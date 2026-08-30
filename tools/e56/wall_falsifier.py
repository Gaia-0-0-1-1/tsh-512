"""E56: distinguish underfitting from post-memorization grokking.

The immutable experiment contract is PREREG.md plus
preregistered_config.json. This runner never writes the repository timeline,
never retries a seed, never stops on test performance, and refuses to
overwrite a run directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "preregistered_config.json"
PREREG_PATH = HERE / "PREREG.md"

sys.path.insert(0, str(ROOT / "tools" / "e6"))
sys.path.insert(0, str(ROOT / "proto"))
sys.path.insert(0, str(ROOT / "tools" / "e20"))

from math_structures import STRUCTURES  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402


HARD_TASKS = (
    "Z4xZ2(Z4xZ2)",
    "Z2x2x2(Z2x2x2)",
    "Z4xZ2(Z2x2x2)",
    "Z2x2x2(Z4xZ2)",
)
MODEL_VARIANT = "tools/e20 TinyTransformer lattice=None bias-free MLP"
SHUFFLE_SEED = 560_001
EXPECTED_CONFIG_SHA256 = (
    "41d4ce49ea1e0e292e09fcd9f6f6c640a2584fbefe1e6acc7324ac0a9e48c2ef"
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_fingerprint(*tensors: torch.Tensor) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode("ascii"))
        h.update(canonical(list(value.shape)).encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def cyc(n: int) -> list[list[int]]:
    return [[(a + b) % n for b in range(n)] for a in range(n)]


def task_tables() -> dict[str, list[list[int]]]:
    return {
        "Z4xZ2": STRUCTURES["Z4xZ2"]["make"](),
        "Z2x2x2": STRUCTURES["Z2x2x2"]["make"](),
        "T8": cyc(8),
    }


def task_spec(name: str) -> tuple[str, str, bool]:
    specs = {
        "Z4xZ2(Z4xZ2)": ("Z4xZ2", "Z4xZ2", False),
        "Z2x2x2(Z2x2x2)": ("Z2x2x2", "Z2x2x2", False),
        "Z4xZ2(Z2x2x2)": ("Z4xZ2", "Z2x2x2", False),
        "Z2x2x2(Z4xZ2)": ("Z2x2x2", "Z4xZ2", False),
        "T8(T8)": ("T8", "T8", False),
        "SHUFFLED_Z4xZ2(Z4xZ2)": ("Z4xZ2", "Z4xZ2", True),
    }
    if name not in specs:
        raise KeyError(f"unknown E56 task: {name}")
    return specs[name]


def full_domain(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    outer_name, inner_name, shuffled = task_spec(name)
    tables = task_tables()
    outer, inner = tables[outer_name], tables[inner_name]
    xs: list[list[int]] = []
    ys: list[int] = []
    for a1 in range(8):
        for b1 in range(8):
            for a2 in range(8):
                for b2 in range(8):
                    c1 = inner[a1][b1]
                    c2 = inner[a2][b2]
                    xs.append([a1, b1, a2, b2])
                    ys.append(outer[c1][c2])
    if shuffled:
        order = random.Random(SHUFFLE_SEED).sample(range(len(ys)), len(ys))
        ys = [ys[i] for i in order]
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def make_data(name: str, split_seed: int, train_fraction: float,
              device: torch.device) -> dict[str, Any]:
    x, y = full_domain(name)
    order = random.Random(split_seed).sample(range(len(x)), len(x))
    cut = int(train_fraction * len(order))
    train_idx = torch.tensor(order[:cut], dtype=torch.long)
    test_idx = torch.tensor(order[cut:], dtype=torch.long)
    label_counts = torch.bincount(y, minlength=8).tolist()
    return {
        "name": name,
        "x": x.to(device),
        "y": y.to(device),
        "train_x": x[train_idx].to(device),
        "train_y": y[train_idx].to(device),
        "test_x": x[test_idx].to(device),
        "test_y": y[test_idx].to(device),
        "fingerprint": tensor_fingerprint(x, y, train_idx, test_idx),
        "full_fingerprint": tensor_fingerprint(x, y),
        "label_counts": label_counts,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    }


class JsonlRecorder:
    def __init__(self, path: Path):
        self.path = path
        self._f = path.open("x", encoding="utf-8", buffering=1)

    def log(self, kind: str, **fields: Any) -> None:
        record = {"kind": kind, **fields}
        self._f.write(canonical(record) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


@dataclass
class SustainedGate:
    required: int
    count: int = 0
    candidate_start: int | None = None
    first_step: int | None = None
    confirmed_step: int | None = None

    def update(self, condition: bool, step: int) -> bool:
        if self.confirmed_step is not None:
            return False
        if condition:
            if self.count == 0:
                self.candidate_start = step
            self.count += 1
            if self.count >= self.required:
                self.first_step = self.candidate_start
                self.confirmed_step = step
                return True
        else:
            self.count = 0
            self.candidate_start = None
        return False


@torch.no_grad()
def evaluate(model: TinyTransformer, data: dict[str, Any]) -> dict[str, float]:
    was_training = model.training
    model.eval()
    train_logits = model(data["train_x"])
    test_logits = model(data["test_x"])
    values = {
        "train_loss": F.cross_entropy(train_logits, data["train_y"]).item(),
        "train_acc": (train_logits.argmax(-1) == data["train_y"])
        .float().mean().item(),
        "test_loss": F.cross_entropy(test_logits, data["test_y"]).item(),
        "test_acc": (test_logits.argmax(-1) == data["test_y"])
        .float().mean().item(),
        "mean_test_logit_norm": test_logits.norm(dim=-1).mean().item(),
        "parameter_l2": math.sqrt(sum(
            p.detach().float().square().sum().item()
            for p in model.parameters()
        )),
    }
    model.train(was_training)
    return values


def seed_model(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(width: int, device: torch.device) -> TinyTransformer:
    return TinyTransformer(8, 8, d=width, lattice=None).to(device)


def make_optimizer(model: TinyTransformer, config: dict[str, Any],
                   weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=weight_decay,
        betas=tuple(config["betas"]),
    )


def make_batch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(56_000_000 + seed)
    return generator


def train_updates(
    *,
    model: TinyTransformer,
    optimizer: torch.optim.AdamW,
    data: dict[str, Any],
    batch_generator: torch.Generator,
    config: dict[str, Any],
    recorder: JsonlRecorder,
    run_key: str,
    phase: str,
    start_step: int,
    updates: int,
    memorize_gate: SustainedGate,
    grok_gate: SustainedGate,
    stop_on_memorization: bool,
) -> tuple[int, dict[str, float]]:
    eval_every = config["eval_every"]
    batch_size = config["batch_size"]
    final_metrics = evaluate(model, data)
    recorder.log("evaluation", run_key=run_key, phase=phase,
                 step=start_step, baseline=True, **final_metrics)
    t0 = time.time()
    model.train()
    last_step = start_step
    for local_step in range(1, updates + 1):
        last_step = start_step + local_step
        idx = torch.randperm(
            data["train_x"].shape[0],
            generator=batch_generator,
            device=data["train_x"].device,
        )[:batch_size]
        logits = model(data["train_x"][idx])
        loss = F.cross_entropy(logits, data["train_y"][idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if local_step % eval_every != 0:
            continue
        final_metrics = evaluate(model, data)
        train_ok = final_metrics["train_acc"] >= config["memorize_threshold"]
        grok_ok = train_ok and (
            final_metrics["test_acc"] >= config["grok_threshold"]
        )
        train_new = memorize_gate.update(train_ok, last_step)
        grok_new = grok_gate.update(grok_ok, last_step)
        recorder.log(
            "evaluation",
            run_key=run_key,
            phase=phase,
            step=last_step,
            baseline=False,
            batch_loss=loss.item(),
            train_gate_count=memorize_gate.count,
            train_gate_new=train_new,
            grok_gate_count=grok_gate.count,
            grok_gate_new=grok_new,
            elapsed_seconds=round(time.time() - t0, 3),
            **final_metrics,
        )
        if stop_on_memorization and train_new:
            break
    return last_step, final_metrics


def classify(memorize: SustainedGate, grok: SustainedGate,
             config: dict[str, Any]) -> str:
    if memorize.confirmed_step is None:
        return "NO_MEMORIZATION_WITHIN_BUDGET"
    if grok.confirmed_step is None:
        return "MEMORIZED_NO_GENERALIZATION"
    window = config["eval_every"] * (config["sustain_evaluations"] - 1)
    assert memorize.first_step is not None and grok.first_step is not None
    if grok.first_step - memorize.first_step <= window:
        return "ORDINARY_GENERALIZATION"
    return "POST_MEMORIZATION_GROK"


def classify_terminal(memorize: SustainedGate, grok: SustainedGate,
                      config: dict[str, Any], metrics: dict[str, float]) -> str:
    status = classify(memorize, grok, config)
    terminal_held = (
        metrics["train_acc"] >= config["memorize_threshold"]
        and metrics["test_acc"] >= config["grok_threshold"]
    )
    if grok.confirmed_step is not None and not terminal_held:
        return status + "_THEN_FORGOT"
    return status


def checkpoint(
    path: Path,
    *,
    model: TinyTransformer,
    optimizer: torch.optim.AdamW,
    batch_generator: torch.Generator,
    metadata: dict[str, Any],
) -> str:
    model_device = next(model.parameters()).device
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "batch_generator_state": batch_generator.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all()
        if model_device.type == "cuda" else [],
        "metadata": metadata,
    }, path)
    return sha256_file(path)


def gate_fields(memorize: SustainedGate, grok: SustainedGate) -> dict[str, Any]:
    delta = None
    if memorize.first_step is not None and grok.first_step is not None:
        delta = grok.first_step - memorize.first_step
    return {
        "memorization_first_step": memorize.first_step,
        "memorization_confirmed_step": memorize.confirmed_step,
        "grok_first_step": grok.first_step,
        "grok_confirmed_step": grok.confirmed_step,
        "delta_grok_steps": delta,
    }


def run_sentinel(
    task: str,
    seed: int,
    data: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    recorder: JsonlRecorder,
) -> dict[str, Any]:
    run_key = f"sentinel::{task}::seed={seed}"
    seed_model(seed)
    model = make_model(config["width"], device)
    optimizer = make_optimizer(model, config, config["sentinel_weight_decay"])
    batch_generator = make_batch_generator(device, seed)
    memorize = SustainedGate(config["sustain_evaluations"])
    grok = SustainedGate(config["sustain_evaluations"])
    final_step, metrics = train_updates(
        model=model,
        optimizer=optimizer,
        data=data,
        batch_generator=batch_generator,
        config=config,
        recorder=recorder,
        run_key=run_key,
        phase="sentinel_wd1_from_scratch",
        start_step=0,
        updates=config["sentinel_steps"],
        memorize_gate=memorize,
        grok_gate=grok,
        stop_on_memorization=False,
    )
    path = run_dir / "checkpoints" / f"sentinel_{safe_name(task)}_s{seed}.pt"
    digest = checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        batch_generator=batch_generator,
        metadata={"run_key": run_key, "step": final_step},
    )
    summary = {
        "run_key": run_key,
        "task": task,
        "seed": seed,
        "phase": "sentinel",
        "classification": classify_terminal(memorize, grok, config, metrics),
        "steps": final_step,
        "checkpoint": path.name,
        "checkpoint_sha256": digest,
        **gate_fields(memorize, grok),
        **metrics,
    }
    recorder.log("cell_summary", **summary)
    return summary


def clone_branch(
    source_model: TinyTransformer,
    source_optimizer: torch.optim.AdamW,
    source_batch_generator: torch.Generator,
    config: dict[str, Any],
    device: torch.device,
    weight_decay: float,
) -> tuple[TinyTransformer, torch.optim.AdamW, torch.Generator]:
    model = make_model(config["width"], device)
    model.load_state_dict(copy.deepcopy(source_model.state_dict()))
    optimizer = make_optimizer(model, config, weight_decay)
    optimizer.load_state_dict(copy.deepcopy(source_optimizer.state_dict()))
    for group in optimizer.param_groups:
        group["weight_decay"] = weight_decay
    generator = make_batch_generator(device, 0)
    generator.set_state(source_batch_generator.get_state().clone())
    return model, optimizer, generator


def run_fork_assay(
    task: str,
    seed: int,
    data: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    recorder: JsonlRecorder,
) -> list[dict[str, Any]]:
    base_key = f"fork::{task}::seed={seed}"
    seed_model(seed)
    model = make_model(config["width"], device)
    optimizer = make_optimizer(model, config, config["phase_a_weight_decay"])
    batch_generator = make_batch_generator(device, seed)
    memorize = SustainedGate(config["sustain_evaluations"])
    grok = SustainedGate(config["sustain_evaluations"])
    phase_a_step, metrics = train_updates(
        model=model,
        optimizer=optimizer,
        data=data,
        batch_generator=batch_generator,
        config=config,
        recorder=recorder,
        run_key=base_key,
        phase="phase_a_wd0_train_gate",
        start_step=0,
        updates=config["phase_a_steps"],
        memorize_gate=memorize,
        grok_gate=grok,
        stop_on_memorization=True,
    )
    if memorize.confirmed_step is None:
        path = run_dir / "checkpoints" / f"phase_a_nomem_{safe_name(task)}_s{seed}.pt"
        digest = checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            batch_generator=batch_generator,
            metadata={"run_key": base_key, "step": phase_a_step},
        )
        summary = {
            "run_key": base_key,
            "task": task,
            "seed": seed,
            "phase": "phase_a",
            "classification": "NO_MEMORIZATION_WITHIN_BUDGET",
            "steps": phase_a_step,
            "checkpoint": path.name,
            "checkpoint_sha256": digest,
            **gate_fields(memorize, grok),
            **metrics,
        }
        recorder.log("cell_summary", **summary)
        return [summary]

    gate_path = run_dir / "checkpoints" / f"gate_{safe_name(task)}_s{seed}.pt"
    gate_digest = checkpoint(
        gate_path,
        model=model,
        optimizer=optimizer,
        batch_generator=batch_generator,
        metadata={"run_key": base_key, "step": phase_a_step},
    )
    recorder.log(
        "gate_checkpoint",
        run_key=base_key,
        task=task,
        seed=seed,
        step=phase_a_step,
        checkpoint=gate_path.name,
        checkpoint_sha256=gate_digest,
        **gate_fields(memorize, grok),
    )

    summaries: list[dict[str, Any]] = []
    for weight_decay in config["branch_weight_decays"]:
        branch_key = f"{base_key}::wd={weight_decay:g}"
        branch_model, branch_optimizer, branch_generator = clone_branch(
            model, optimizer, batch_generator, config, device, weight_decay
        )
        branch_memorize = copy.deepcopy(memorize)
        # A branch must establish its own post-fork sustained test gate. The
        # pre-fork gate is recorded separately and is never inherited.
        branch_grok = SustainedGate(config["sustain_evaluations"])
        final_step, final_metrics = train_updates(
            model=branch_model,
            optimizer=branch_optimizer,
            data=data,
            batch_generator=branch_generator,
            config=config,
            recorder=recorder,
            run_key=branch_key,
            phase=f"post_gate_wd_{weight_decay:g}",
            start_step=phase_a_step,
            updates=config["post_gate_steps"],
            memorize_gate=branch_memorize,
            grok_gate=branch_grok,
            stop_on_memorization=False,
        )
        path = run_dir / "checkpoints" / (
            f"final_{safe_name(task)}_s{seed}_wd{weight_decay:g}.pt"
        )
        digest = checkpoint(
            path,
            model=branch_model,
            optimizer=branch_optimizer,
            batch_generator=branch_generator,
            metadata={"run_key": branch_key, "step": final_step},
        )
        terminal_held = (
            final_metrics["train_acc"] >= config["memorize_threshold"]
            and final_metrics["test_acc"] >= config["grok_threshold"]
        )
        if grok.confirmed_step is not None:
            branch_classification = (
                "PRE_FORK_ORDINARY_GENERALIZATION_RETAINED"
                if terminal_held
                else "PRE_FORK_ORDINARY_GENERALIZATION_THEN_FORGOT"
            )
        else:
            branch_classification = classify_terminal(
                branch_memorize, branch_grok, config, final_metrics
            )
        summary = {
            "run_key": branch_key,
            "task": task,
            "seed": seed,
            "phase": "post_gate",
            "weight_decay": weight_decay,
            "classification": branch_classification,
            "pre_fork_grok_first_step": grok.first_step,
            "pre_fork_grok_confirmed_step": grok.confirmed_step,
            "terminal_gate_held": terminal_held,
            "steps": final_step,
            "checkpoint": path.name,
            "checkpoint_sha256": digest,
            **gate_fields(branch_memorize, branch_grok),
            **final_metrics,
        }
        recorder.log("cell_summary", **summary)
        summaries.append(summary)
        del branch_model, branch_optimizer
    return summaries


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_")


def environment_record(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    driver_version = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()[0].strip()
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": driver_version,
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device))
        if device.type == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(device),
        "gpu": gpu,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "model_variant": MODEL_VARIANT,
        "git_head": git_head,
        "git_status": git_status,
        "timeline_sha256": sha256_file(ROOT / "timeline.jsonl"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON receipt after flushing its bytes."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def make_manifest(run_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        entries.append({
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"files": entries}


def aggregate_verdict(summaries: list[dict[str, Any]],
                      config: dict[str, Any]) -> dict[str, Any]:
    """Score paired task families and the two preregistered controls."""
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    sentinel: dict[tuple[str, int], dict[str, Any]] = {}
    for summary in summaries:
        key = (summary["task"], summary["seed"])
        if summary["phase"] == "sentinel":
            sentinel[key] = summary
        elif summary["run_key"].startswith("fork::"):
            cells.setdefault(key, []).append(summary)

    def cell_memorized(task: str, seed: int) -> bool:
        return any(
            row["memorization_confirmed_step"] is not None
            for row in cells.get((task, seed), [])
        )

    def cell_generalized(task: str, seed: int) -> bool:
        return any(
            row["grok_confirmed_step"] is not None
            or row.get("pre_fork_grok_confirmed_step") is not None
            for row in cells.get((task, seed), [])
        )

    seeds = config["seeds"]
    missing_cells = [
        f"{task}::seed={seed}"
        for task in HARD_TASKS
        for seed in seeds
        if (task, seed) not in cells
    ]
    missing_fork_cells = [
        f"{task}::seed={seed}"
        for task in config["tasks"]
        for seed in seeds
        if (task, seed) not in cells
    ]
    missing_sentinel_cells = [
        f"{task}::seed={seed}"
        for task in config["sentinel_tasks"]
        for seed in seeds
        if (task, seed) not in sentinel
    ]
    hard_by_family = {}
    for task in HARD_TASKS:
        memorized = [cell_memorized(task, seed) for seed in seeds]
        generalized = [cell_generalized(task, seed) for seed in seeds]
        hard_by_family[task] = {
            "memorized_by_seed": memorized,
            "generalized_by_seed": generalized,
            "both_seeds_no_memorization": not any(memorized),
            "both_seeds_memorized": all(memorized),
        }

    no_mem_families = sum(
        row["both_seeds_no_memorization"] for row in hard_by_family.values()
    )
    memorized_families = sum(
        row["both_seeds_memorized"] for row in hard_by_family.values()
    )
    t8_memorized = [cell_memorized("T8(T8)", seed) for seed in seeds]
    t8_generalized = [cell_generalized("T8(T8)", seed) for seed in seeds]
    shuffle_generalized = [
        cell_generalized("SHUFFLED_Z4xZ2(Z4xZ2)", seed) for seed in seeds
    ]
    shuffle_invalid = any(shuffle_generalized)
    floor_pass = all(t8_memorized)

    if config.get("_smoke"):
        score = "NON_EVIDENTIARY_SMOKE"
    elif missing_fork_cells or missing_sentinel_cells:
        score = "INCOMPLETE"
    elif shuffle_invalid:
        score = "INVALID_SHUFFLED_CONTROL"
    elif not floor_pass:
        score = "OPEN_STRUCTURED_FLOOR_CONTROL"
    elif no_mem_families >= 3:
        score = "PREDICTION_HIT_UNDERFITTING_AT_CONFIG"
    elif memorized_families >= 3:
        score = "PREDICTION_MISS_HARD_FAMILIES_MEMORIZE"
    else:
        score = "OPEN_MIXED_PILOT"

    return {
        "score": score,
        "missing_hard_cells": missing_cells,
        "missing_fork_cells": missing_fork_cells,
        "missing_sentinel_cells": missing_sentinel_cells,
        "hard_by_family": hard_by_family,
        "hard_families_both_seeds_no_memorization": no_mem_families,
        "hard_families_both_seeds_memorized": memorized_families,
        "structured_control": {
            "memorized_by_seed": t8_memorized,
            "generalized_by_seed": t8_generalized,
            "floor_pass": floor_pass,
        },
        "shuffled_control": {
            "generalized_by_seed": shuffle_generalized,
            "invalidates_assay": shuffle_invalid,
        },
        "sentinel": {
            f"{task}::seed={seed}": row["classification"]
            for (task, seed), row in sorted(sentinel.items())
        },
        "scope": "two paired seeds; legacy random point split; not structural OOD",
    }


def snapshot_sources(run_dir: Path) -> dict[str, str]:
    """Copy every directly used source/config file into the immutable run."""
    source_dir = run_dir / "source"
    source_dir.mkdir()
    sources = {
        "PREREG.md": PREREG_PATH,
        "preregistered_config.json": CONFIG_PATH,
        "wall_falsifier.py": Path(__file__),
        "test_wall_falsifier.py": HERE / "test_wall_falsifier.py",
        "hyperbyte_test.py": ROOT / "tools" / "e20" / "hyperbyte_test.py",
        "math_structures.py": ROOT / "tools" / "e6" / "math_structures.py",
    }
    receipts: dict[str, str] = {}
    for name, source in sources.items():
        target = source_dir / name
        shutil.copy2(source, target)
        receipts[name] = sha256_file(target)
    return receipts


def load_config(smoke: bool) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if smoke:
        config = copy.deepcopy(config)
        config["_smoke"] = True
        config["tasks"] = ["T8(T8)"]
        config["sentinel_tasks"] = [HARD_TASKS[0]]
        config["seeds"] = [0]
        config["sentinel_steps"] = 400
        config["phase_a_steps"] = 4000
        config["post_gate_steps"] = 400
        config["eval_every"] = 100
        config["sustain_evaluations"] = 2
    return config


def validate_config(config: dict[str, Any], smoke: bool) -> None:
    if not smoke:
        observed = sha256_file(CONFIG_PATH)
        if observed != EXPECTED_CONFIG_SHA256:
            raise RuntimeError(
                "evidence config digest differs from the runner's registered digest: "
                f"expected {EXPECTED_CONFIG_SHA256}, observed {observed}"
            )
    if config["device"] != "cuda":
        raise RuntimeError("E56 is preregistered CUDA-only while E55 is active")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; failing closed to avoid CPU contention")
    if config["sustain_evaluations"] < 2:
        raise ValueError("sustained gates require at least two evaluations")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="non-evidentiary 400-step implementation smoke test")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    config = load_config(args.smoke)
    validate_config(config, args.smoke)
    device = torch.device("cuda:0")

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    source_receipts = snapshot_sources(run_dir)
    results_path = run_dir / "results.jsonl"
    recorder = JsonlRecorder(results_path)
    summaries: list[dict[str, Any]] = []
    started = time.time()

    try:
        recorder.log(
            "run_start",
            evidence=not args.smoke,
            config=config,
            config_fingerprint=sha256_bytes(canonical(config).encode("utf-8")),
            environment=environment_record(device),
            source_receipts=source_receipts,
            model_parameters=sum(
                p.numel() for p in make_model(config["width"], device).parameters()
            ),
        )

        # First diagnose the E47/E20 model/data/wd condition while logging the
        # previously missing train trajectory. This is deliberately not called
        # an exact CPU/50k stochastic replay.
        for task in config["sentinel_tasks"]:
            for seed in config["seeds"]:
                data = make_data(task, seed, config["train_fraction"], device)
                recorder.log(
                    "dataset",
                    task=task,
                    seed=seed,
                    fingerprint=data["fingerprint"],
                    full_fingerprint=data["full_fingerprint"],
                    label_counts=data["label_counts"],
                    train_size=data["train_size"],
                    test_size=data["test_size"],
                    split_kind="legacy_random_point_split_not_structural_ood",
                )
                summaries.append(run_sentinel(
                    task, seed, data, config, device, run_dir, recorder
                ))
                print(canonical(summaries[-1]), flush=True)
                write_json(run_dir / "summary.partial.json", summaries)

        # Then ask whether wd=0 can reach a sustained memorization gate and,
        # only when it can, fork matched post-gate continuations.
        for task in config["tasks"]:
            for seed in config["seeds"]:
                data = make_data(task, seed, config["train_fraction"], device)
                recorder.log(
                    "dataset",
                    task=task,
                    seed=seed,
                    fingerprint=data["fingerprint"],
                    full_fingerprint=data["full_fingerprint"],
                    label_counts=data["label_counts"],
                    train_size=data["train_size"],
                    test_size=data["test_size"],
                    split_kind="legacy_random_point_split_not_structural_ood",
                )
                new_summaries = run_fork_assay(
                    task, seed, data, config, device, run_dir, recorder
                )
                summaries.extend(new_summaries)
                for summary in new_summaries:
                    print(canonical(summary), flush=True)
                write_json(run_dir / "summary.partial.json", summaries)

        verdict = aggregate_verdict(summaries, config)
        write_json(run_dir / "verdict.json", verdict)
        recorder.log("aggregate_verdict", **verdict)
        recorder.log("run_complete", cells=len(summaries), verdict=verdict,
                     elapsed_seconds=round(time.time() - started, 3))
    except BaseException as exc:
        recorder.log("run_interrupted", error=repr(exc),
                     elapsed_seconds=round(time.time() - started, 3))
        raise
    finally:
        recorder.close()
        write_json(run_dir / "summary.json", summaries)
        write_json(run_dir / "manifest.json", make_manifest(run_dir))

    print(f"E56 complete: {len(summaries)} terminal cells in "
          f"{(time.time() - started) / 60:.1f} min")
    print(f"results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
