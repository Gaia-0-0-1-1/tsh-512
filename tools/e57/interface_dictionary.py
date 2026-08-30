"""E57 confirmatory replay: break and repair a categorical join dictionary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "preregistered_config.json"
PREREG_PATH = HERE / "PREREG.md"
PREREGISTRATION_PATH = HERE / "preregistration.json"
REGISTRATION_PATH = HERE / "registration.json"
EXPECTED_CONFIG_SHA256 = (
    "51e26cab5333be2c971849ff77b064b1b9471a65802e49f1e0d709768aefad55"
)

sys.path.insert(0, str(ROOT / "tools" / "e6"))
sys.path.insert(0, str(ROOT / "proto"))
sys.path.insert(0, str(ROOT / "tools" / "e20"))
sys.path.insert(0, str(ROOT / "tools" / "e36"))

from math_structures import STRUCTURES, make_task  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402
from h4_fusion import b_stack, composite_domain, soft_fused_logits  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(canonical(list(value.shape)).encode("ascii"))
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_mapping_fingerprint(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(canonical(list(value.shape)).encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_dict_fingerprint(model: torch.nn.Module) -> str:
    return state_mapping_fingerprint(model.state_dict())


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def write_text(path: Path, value: str, encoding: str = "utf-8") -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding=encoding) as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class ChainedRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.seq = 0
        self.prev = "0" * 64
        self._f = path.open("x", encoding="utf-8", buffering=1)

    def log(self, kind: str, **fields: Any) -> dict[str, Any]:
        unsigned = {
            "seq": self.seq,
            "prev": self.prev,
            "kind": kind,
            **fields,
        }
        digest = hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest()
        record = {**unsigned, "hash": digest}
        self._f.write(canonical(record) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self.prev = digest
        self.seq += 1
        return record

    def close(self) -> None:
        self._f.close()


def verify_result_chain(path: Path) -> dict[str, Any]:
    expected_prev = "0" * 64
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            digest = record.pop("hash")
            if record["seq"] != count or record["prev"] != expected_prev:
                raise RuntimeError(f"result chain link mismatch at row {count}")
            observed = hashlib.sha256(
                canonical(record).encode("utf-8")
            ).hexdigest()
            if observed != digest:
                raise RuntimeError(f"result chain hash mismatch at row {count}")
            expected_prev = digest
            count += 1
    return {"records": count, "head": expected_prev}


def full_component_domain(task: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = STRUCTURES[task]["n"]
    table = STRUCTURES[task]["make"]()
    x = torch.tensor([[a, b] for a in range(n) for b in range(n)],
                     dtype=torch.long)
    y = torch.tensor([table[a][b] for a in range(n) for b in range(n)],
                     dtype=torch.long)
    return x, y


def train_fixed_component(task: str, config: dict[str, Any],
                          recorder: ChainedRecorder,
                          component_dir: Path) -> tuple[TinyTransformer, dict[str, Any]]:
    seed = config["seed"]
    steps = config["component_steps"][task]
    ds = make_task(task, config["train_fraction"], seed, "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    model = TinyTransformer(
        STRUCTURES[task]["n"], STRUCTURES[task]["n"],
        d=config["width"], lattice=config["lattice"],
    )
    # Reproduce the post-initialization CPU RNG stream without letting any
    # later instrumentation consume it.
    batch_generator = torch.Generator(device="cpu")
    batch_generator.set_state(torch.get_rng_state().clone())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=tuple(config["betas"]),
    )
    x_train, y_train = ds["train_x"], ds["train_y"]
    started = time.time()
    last_loss = None
    model.train()
    for _step in range(1, steps + 1):
        idx = torch.randperm(
            x_train.shape[0], generator=batch_generator
        )[:config["batch_size"]]
        logits = model(x_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

    full_x, full_y = full_component_domain(task)
    model.eval()
    with torch.no_grad():
        full_logits = model(full_x)
        full_pred = full_logits.argmax(-1)
        train_acc = (model(ds["train_x"]).argmax(-1) == ds["train_y"]).float().mean().item()
        test_acc = (model(ds["test_x"]).argmax(-1) == ds["test_y"]).float().mean().item()
    full_correct = int((full_pred == full_y).sum().item())
    path = component_dir / f"{task}.pt"
    torch.save({
        "task": task,
        "seed": seed,
        "steps": steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "batch_generator_state": batch_generator.get_state(),
        "full_x_fingerprint": tensor_fingerprint(full_x),
        "full_y_fingerprint": tensor_fingerprint(full_y),
    }, path)
    summary = {
        "task": task,
        "seed": seed,
        "steps": steps,
        "terminal_batch_loss": last_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "full_correct": full_correct,
        "full_total": len(full_y),
        "admitted": full_correct == len(full_y),
        "seconds": round(time.time() - started, 3),
        "checkpoint": path.name,
        "checkpoint_sha256": sha256_file(path),
        "state_fingerprint": state_dict_fingerprint(model),
        "prediction_fingerprint": tensor_fingerprint(full_pred),
    }
    recorder.log("component", **summary)
    print(canonical(summary), flush=True)
    return model, summary


def permutation_tensors(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    pi = torch.tensor(config["permutation_old_to_new"], dtype=torch.long)
    if sorted(pi.tolist()) != list(range(8)):
        raise RuntimeError("registered permutation is not bijective")
    if torch.any(pi == torch.arange(8)):
        raise RuntimeError("registered permutation is not a derangement")
    inv = torch.empty_like(pi)
    inv[pi] = torch.arange(8)
    if not torch.equal(inv[pi], torch.arange(8)):
        raise RuntimeError("inverse permutation identity failed")
    return pi, inv


def joined_logits(inner: TinyTransformer, outer: TinyTransformer,
                  x: torch.Tensor, temperature: float,
                  pi: torch.Tensor, inv: torch.Tensor) -> dict[str, torch.Tensor]:
    l1 = inner(x[:, :2])
    l2 = inner(x[:, 2:])
    p1 = torch.softmax(l1 / temperature, dim=-1)
    p2 = torch.softmax(l2 / temperature, dim=-1)
    q1, q2 = p1[:, inv], p2[:, inv]
    embeddings = outer.embed.weight

    def enter(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        return b_stack(outer, torch.stack([e1, e2], dim=1))

    return {
        "baseline": enter(p1 @ embeddings, p2 @ embeddings),
        "wrong": enter(q1 @ embeddings, q2 @ embeddings),
        "repair": enter(q1[:, pi] @ embeddings, q2[:, pi] @ embeddings),
        "gauge": enter(q1 @ embeddings[inv], q2 @ embeddings[inv]),
    }


def assay_pair(outer_name: str, inner_name: str,
               components: dict[str, TinyTransformer],
               config: dict[str, Any], recorder: ChainedRecorder,
               arrays_dir: Path) -> tuple[
                   list[dict[str, Any]], dict[str, list[str]]
               ]:
    outer, inner = components[outer_name], components[inner_name]
    ds = composite_domain(outer_name, inner_name, seed=config["seed"])
    x, truth = ds["x"], ds["y"]
    if tuple(x.shape) != (4096, 4) or tuple(truth.shape) != (4096,):
        raise RuntimeError("composite domain shape mismatch")
    counts = torch.bincount(truth, minlength=8).tolist()
    if counts != [512] * 8:
        raise RuntimeError(f"composite labels are not balanced: {counts}")
    pi, inv = permutation_tensors(config)
    rows: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {
        "core": [],
        "gauge": [],
        "break": [],
    }
    raw: dict[str, Any] = {
        "pair": f"{outer_name}({inner_name})",
        "outer": outer_name,
        "inner": inner_name,
        "x": x,
        "truth": truth,
        "x_fingerprint": tensor_fingerprint(x),
        "truth_fingerprint": tensor_fingerprint(truth),
        "arms": {},
    }
    pair = f"{outer_name}({inner_name})"
    exact_temps = set(config["exact_truth_temperatures"])
    with torch.no_grad():
        for temperature in config["temperatures"]:
            arms = joined_logits(inner, outer, x, temperature, pi, inv)
            reference = soft_fused_logits(inner, outer, x, temperature)
            if not torch.equal(arms["baseline"], reference):
                failures["core"].append(
                    f"{pair}@T={temperature}: baseline/E36 mismatch"
                )
            preds = {name: logits.argmax(-1) for name, logits in arms.items()}
            acc = {
                name: (pred == truth).float().mean().item()
                for name, pred in preds.items()
            }
            mismatch = {
                name: int((pred != truth).sum().item())
                for name, pred in preds.items()
            }
            repair_equal = torch.equal(arms["repair"], arms["baseline"])
            gauge_close = torch.allclose(
                arms["gauge"], arms["baseline"],
                rtol=config["rtol"], atol=config["atol"],
            )
            gauge_max_diff = (
                arms["gauge"] - arms["baseline"]
            ).abs().max().item()
            if not repair_equal:
                failures["core"].append(
                    f"{pair}@T={temperature}: repair not bit-identical"
                )
            if not gauge_close:
                failures["gauge"].append(
                    f"{pair}@T={temperature}: gauge numerical drift"
                )
            if temperature in exact_temps:
                for arm in ("baseline", "repair", "gauge"):
                    if mismatch[arm] != 0:
                        bucket = "gauge" if arm == "gauge" else "core"
                        failures[bucket].append(
                            f"{pair}@T={temperature}: {arm} not exact"
                        )
            if temperature == 1.0 and mismatch["wrong"] == 0:
                failures["break"].append(
                    f"{pair}@T=1: wrong dictionary remained exact"
                )

            row = {
                "pair": pair,
                "outer": outer_name,
                "inner": inner_name,
                "temperature": temperature,
                "accuracy": acc,
                "mismatches": mismatch,
                "repair_bit_identical": repair_equal,
                "gauge_close": gauge_close,
                "gauge_max_abs_logit_diff": gauge_max_diff,
                "prediction_fingerprints": {
                    name: tensor_fingerprint(pred) for name, pred in preds.items()
                },
                "logit_fingerprints": {
                    name: tensor_fingerprint(logits) for name, logits in arms.items()
                },
            }
            rows.append(row)
            recorder.log("assay", **row)
            raw["arms"][str(temperature)] = {
                name: {"logits": arms[name], "pred": preds[name]}
                for name in arms
            }
    array_path = arrays_dir / f"{outer_name}_of_{inner_name}.pt"
    torch.save(raw, array_path)
    recorder.log(
        "array_artifact", pair=pair, path=array_path.name,
        sha256=sha256_file(array_path), bytes=array_path.stat().st_size,
    )
    return rows, failures


def instrument_checks(components: dict[str, TinyTransformer],
                      recorder: ChainedRecorder) -> list[str]:
    failures: list[str] = []
    for task, model in components.items():
        full_x, _ = full_component_domain(task)
        with torch.no_grad():
            direct = model(full_x)
            entered = b_stack(model, model.embed(full_x))
        max_diff = (direct - entered).abs().max().item()
        passed = max_diff < 1e-5
        recorder.log("b_stack_faithfulness", task=task, passed=passed,
                     max_abs_diff=max_diff)
        if not passed:
            failures.append(f"{task}: b_stack max diff {max_diff}")
    return failures


def environment_record(config: dict[str, Any]) -> dict[str, Any]:
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_build_config": torch.__config__.show(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "git_head": git_head,
        "git_status": git_status,
        "timeline_sha256": sha256_file(ROOT / "timeline.jsonl"),
        "config_fingerprint": hashlib.sha256(
            canonical(config).encode("utf-8")
        ).hexdigest(),
    }


def snapshot_sources(run_dir: Path) -> dict[str, str]:
    source_dir = run_dir / "source"
    source_dir.mkdir()
    sources = {
        "PREREG.md": PREREG_PATH,
        "preregistered_config.json": CONFIG_PATH,
        "preregistration.json": PREREGISTRATION_PATH,
        "registration.json": REGISTRATION_PATH,
        "interface_dictionary.py": Path(__file__),
        "test_interface_dictionary.py": HERE / "test_interface_dictionary.py",
        "h4_fusion.py": ROOT / "tools" / "e36" / "h4_fusion.py",
        "hyperbyte_test.py": ROOT / "tools" / "e20" / "hyperbyte_test.py",
        "math_structures.py": ROOT / "tools" / "e6" / "math_structures.py",
    }
    receipts = {}
    for name, source in sources.items():
        target = source_dir / name
        shutil.copy2(source, target)
        receipts[name] = sha256_file(target)
    return receipts


def make_manifest(run_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name in {"manifest.json", "manifest.sha256"}:
            continue
        files.append({
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"files": files}


def validate_config() -> dict[str, Any]:
    observed = sha256_file(CONFIG_PATH)
    if observed != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            f"config digest mismatch: expected {EXPECTED_CONFIG_SHA256}, "
            f"observed {observed}"
        )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    permutation_tensors(config)
    return config


def verify_registered_hashes(
    document_path: Path, expected_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Fail closed if a registration's exact path/hash map has drifted."""
    document = json.loads(document_path.read_text(encoding="utf-8"))
    expected_hashes = document.get("sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise RuntimeError(f"{document_path.name}: missing sha256 map")
    registered_paths = set(expected_hashes)
    if expected_paths is not None and registered_paths != expected_paths:
        missing = sorted(expected_paths - registered_paths)
        extra = sorted(registered_paths - expected_paths)
        raise RuntimeError(
            f"{document_path.name}: registered path-set mismatch; "
            f"missing={missing}, extra={extra}"
        )

    failures = []
    receipts = {}
    root = ROOT.resolve()
    for relative, expected in sorted(expected_hashes.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute():
            failures.append(f"absolute registered path: {relative}")
            continue
        path = (ROOT / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"registered path escapes repository: {relative}")
            continue
        if not path.is_file():
            failures.append(f"registered path missing: {relative}")
            continue
        observed = sha256_file(path)
        receipts[relative] = observed
        if observed != expected:
            failures.append(
                f"registered hash mismatch for {relative}: "
                f"expected {expected}, observed {observed}"
            )
    if failures:
        raise RuntimeError("; ".join(failures))
    return {
        "document": document_path.name,
        "document_sha256": sha256_file(document_path),
        "files": receipts,
    }


def timeline_records() -> list[dict[str, Any]]:
    records = []
    expected_prev = "0" * 64
    expected_seq = 1
    with (ROOT / "timeline.jsonl").open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            record = json.loads(line)
            digest = record.pop("hash")
            if (
                record.get("seq") != expected_seq
                or record.get("prev") != expected_prev
            ):
                raise RuntimeError(f"timeline link mismatch at row {row_index}")
            observed = hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()
            if observed != digest:
                raise RuntimeError(f"timeline hash mismatch at row {row_index}")
            records.append({**record, "hash": digest})
            expected_prev = digest
            expected_seq += 1
    return records


def validate_execution_registration() -> dict[str, Any]:
    """Validate both registration documents and their timeline anchors."""
    if not REGISTRATION_PATH.is_file():
        raise RuntimeError("execution registration is missing")
    preregistration = json.loads(
        PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    for name, document in (
        ("preregistration", preregistration),
        ("registration", registration),
    ):
        if document.get("base_commit") != head:
            raise RuntimeError(
                f"{name} base commit mismatch: registered "
                f"{document.get('base_commit')}, observed {head}"
            )

    prereg_paths = {
        "tools/e57/PREREG.md",
        "tools/e57/preregistered_config.json",
        "tools/e36/h4_fusion.py",
        "tools/e20/hyperbyte_test.py",
        "tools/e6/math_structures.py",
    }
    execution_paths = prereg_paths | {
        "tools/e57/preregistration.json",
        "tools/e57/interface_dictionary.py",
        "tools/e57/test_interface_dictionary.py",
    }
    prereg_receipt = verify_registered_hashes(
        PREREGISTRATION_PATH, prereg_paths
    )
    registration_receipt = verify_registered_hashes(
        REGISTRATION_PATH, execution_paths
    )
    if registration.get("preregistration_sha256") != sha256_file(
        PREREGISTRATION_PATH
    ):
        raise RuntimeError("registration does not bind preregistration.json")

    records = timeline_records()
    anchor = registration.get("preregistration_timeline_anchor", {})
    preregistration_sha = sha256_file(PREREGISTRATION_PATH)
    if (
        anchor.get("subject") != "e57/interface-dictionary-prereg"
        or anchor.get("payload_sha256") != preregistration_sha
    ):
        raise RuntimeError("registration names the wrong preregistration anchor")
    anchor_seq = anchor.get("seq")
    anchor_matches = [
        record for record in records if record.get("seq") == anchor_seq
    ]
    if not isinstance(anchor_seq, int) or len(anchor_matches) != 1:
        raise RuntimeError("invalid preregistration timeline anchor")
    anchored = anchor_matches[0]
    for field in ("hash", "subject", "payload_sha256"):
        if anchored.get(field) != anchor.get(field):
            raise RuntimeError(
                f"preregistration timeline anchor mismatch for {field}"
            )
    if anchored.get("kind") != "prediction":
        raise RuntimeError("preregistration timeline anchor is not a prediction")

    subject = registration.get("execution_timeline_subject")
    if subject != "e57/interface-dictionary-execution-registration":
        raise RuntimeError("unexpected execution timeline subject")
    registration_sha = sha256_file(REGISTRATION_PATH)
    execution_anchors = [
        record for record in records
        if record.get("subject") == subject
        and record.get("payload_sha256") == registration_sha
    ]
    if len(execution_anchors) != 1:
        raise RuntimeError(
            "execution registration must have exactly one matching timeline anchor"
        )
    execution_anchor = execution_anchors[0]
    if (
        execution_anchor.get("kind") != "prediction"
        or execution_anchor.get("seq", -1) <= anchored.get("seq", -1)
    ):
        raise RuntimeError("execution timeline anchor has invalid kind or order")
    return {
        "preregistration": prereg_receipt,
        "registration": registration_receipt,
        "preregistration_timeline_anchor": anchored,
        "execution_timeline_anchor": execution_anchor,
    }


def validate_snapshot_against_registration(
    source_receipts: dict[str, str],
    registration_receipts: dict[str, Any],
) -> None:
    expected = {
        Path(relative).name: digest
        for relative, digest in registration_receipts["registration"][
            "files"
        ].items()
    }
    expected["registration.json"] = registration_receipts["registration"][
        "document_sha256"
    ]
    if source_receipts != expected:
        raise RuntimeError(
            "source snapshots do not exactly match execution registration"
        )


def reduce_integrity(
    run_dir: Path,
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
    assay_rows: list[dict[str, Any]],
    source_receipts: dict[str, str],
    state_before: dict[str, str],
    state_after: dict[str, str],
) -> list[str]:
    """Reduce exact cell/artifact/state completeness before any PASS."""
    failures: list[str] = []
    expected_tasks = set(config["component_steps"])
    observed_tasks = [row["task"] for row in summaries]
    if len(observed_tasks) != len(expected_tasks) or set(observed_tasks) != expected_tasks:
        failures.append(
            f"component key mismatch: expected={sorted(expected_tasks)}, "
            f"observed={observed_tasks}"
        )

    expected_cells = {
        (outer, inner, float(temperature))
        for outer, inner in config["pairs"]
        for temperature in config["temperatures"]
    }
    observed_cells = [
        (row["outer"], row["inner"], float(row["temperature"]))
        for row in assay_rows
    ]
    if len(observed_cells) != len(expected_cells) or set(observed_cells) != expected_cells:
        failures.append("assay cell identity mismatch")

    expected_sources = {
        "PREREG.md",
        "preregistered_config.json",
        "preregistration.json",
        "registration.json",
        "interface_dictionary.py",
        "test_interface_dictionary.py",
        "h4_fusion.py",
        "hyperbyte_test.py",
        "math_structures.py",
    }
    source_dir = run_dir / "source"
    observed_sources = {
        path.name for path in source_dir.iterdir() if path.is_file()
    }
    if set(source_receipts) != expected_sources or observed_sources != expected_sources:
        failures.append("source snapshot set mismatch")
    for name in sorted(expected_sources & observed_sources & set(source_receipts)):
        if sha256_file(source_dir / name) != source_receipts[name]:
            failures.append(f"source snapshot hash mismatch: {name}")

    component_dir = run_dir / "components"
    expected_checkpoints = {f"{task}.pt" for task in expected_tasks}
    observed_checkpoints = {
        path.name for path in component_dir.iterdir() if path.is_file()
    }
    if observed_checkpoints != expected_checkpoints:
        failures.append("component checkpoint set mismatch")
    summaries_by_task = {row["task"]: row for row in summaries}
    for task in sorted(expected_tasks):
        path = component_dir / f"{task}.pt"
        summary = summaries_by_task.get(task)
        if not path.is_file() or summary is None:
            continue
        if sha256_file(path) != summary["checkpoint_sha256"]:
            failures.append(f"component checkpoint hash mismatch: {task}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_fingerprint = state_mapping_fingerprint(checkpoint["model"])
        joined_fingerprints = {
            checkpoint_fingerprint,
            summary["state_fingerprint"],
            state_before.get(task),
            state_after.get(task),
        }
        if None in joined_fingerprints or len(joined_fingerprints) != 1:
            failures.append(f"component checkpoint state mismatch: {task}")

    if state_before != state_after or set(state_before) != expected_tasks:
        failures.append("component state changed during frozen assay")

    expected_array_metadata = {
        f"{outer}_of_{inner}.pt": (outer, inner, f"{outer}({inner})")
        for outer, inner in config["pairs"]
    }
    expected_arrays = set(expected_array_metadata)
    rows_by_cell = {
        (row["outer"], row["inner"], float(row["temperature"])): row
        for row in assay_rows
    }
    array_dir = run_dir / "arrays"
    observed_arrays = {
        path.name for path in array_dir.iterdir() if path.is_file()
    }
    if observed_arrays != expected_arrays:
        failures.append("array artifact set mismatch")
    expected_temperature_keys = {str(value) for value in config["temperatures"]}
    for name in sorted(expected_arrays & observed_arrays):
        raw = torch.load(array_dir / name, map_location="cpu", weights_only=True)
        expected_outer, expected_inner, expected_pair = expected_array_metadata[name]
        if (
            raw.get("outer") != expected_outer
            or raw.get("inner") != expected_inner
            or raw.get("pair") != expected_pair
        ):
            failures.append(f"array pair metadata mismatch: {name}")
        expected_domain = composite_domain(
            expected_outer, expected_inner, seed=config["seed"]
        )
        if (
            not torch.equal(raw["x"], expected_domain["x"])
            or not torch.equal(raw["truth"], expected_domain["y"])
        ):
            failures.append(f"array domain content mismatch: {name}")
        if raw.get("x_fingerprint") != tensor_fingerprint(raw["x"]):
            failures.append(f"array x fingerprint mismatch: {name}")
        if raw.get("truth_fingerprint") != tensor_fingerprint(raw["truth"]):
            failures.append(f"array truth fingerprint mismatch: {name}")
        if set(raw.get("arms", {})) != expected_temperature_keys:
            failures.append(f"array temperature set mismatch: {name}")
            continue
        for temperature, arms in raw["arms"].items():
            row = rows_by_cell.get(
                (raw.get("outer"), raw.get("inner"), float(temperature))
            )
            if row is None:
                failures.append(f"array has no assay row: {name}@{temperature}")
                continue
            if set(arms) != {"baseline", "wrong", "repair", "gauge"}:
                failures.append(f"array arm set mismatch: {name}@{temperature}")
                continue
            for arm, tensors in arms.items():
                if tuple(tensors["logits"].shape) != (4096, 8):
                    failures.append(
                        f"array logit shape mismatch: {name}@{temperature}/{arm}"
                    )
                if tuple(tensors["pred"].shape) != (4096,):
                    failures.append(
                        f"array pred shape mismatch: {name}@{temperature}/{arm}"
                    )
                pred = tensors["pred"]
                logits = tensors["logits"]
                if not torch.equal(pred, logits.argmax(-1)):
                    failures.append(
                        f"array pred/logit mismatch: {name}@{temperature}/{arm}"
                    )
                if tensor_fingerprint(pred) != row["prediction_fingerprints"][arm]:
                    failures.append(
                        f"array pred fingerprint mismatch: {name}@{temperature}/{arm}"
                    )
                if tensor_fingerprint(logits) != row["logit_fingerprints"][arm]:
                    failures.append(
                        f"array logit fingerprint mismatch: {name}@{temperature}/{arm}"
                    )
                accuracy = (pred == raw["truth"]).float().mean().item()
                mismatches = int((pred != raw["truth"]).sum().item())
                if accuracy != row["accuracy"][arm] or mismatches != row[
                    "mismatches"
                ][arm]:
                    failures.append(
                        f"array metric mismatch: {name}@{temperature}/{arm}"
                    )
            repair_equal = torch.equal(
                arms["repair"]["logits"], arms["baseline"]["logits"]
            )
            gauge_close = torch.allclose(
                arms["gauge"]["logits"],
                arms["baseline"]["logits"],
                rtol=config["rtol"],
                atol=config["atol"],
            )
            gauge_diff = (
                arms["gauge"]["logits"] - arms["baseline"]["logits"]
            ).abs().max().item()
            if (
                repair_equal != row["repair_bit_identical"]
                or gauge_close != row["gauge_close"]
                or gauge_diff != row["gauge_max_abs_logit_diff"]
            ):
                failures.append(f"array identity metric mismatch: {name}@{temperature}")

    verify_result_chain(run_dir / "results.jsonl")
    with (run_dir / "results.jsonl").open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assay_records = []
    for record in records:
        if record["kind"] != "assay":
            continue
        assay_records.append({
            key: value
            for key, value in record.items()
            if key not in {"seq", "prev", "hash", "kind"}
        })
    if (
        len(assay_records) != len(assay_rows)
        or [canonical(row) for row in assay_records]
        != [canonical(row) for row in assay_rows]
    ):
        failures.append("assay result-chain rows do not match summary rows")

    array_records = [row for row in records if row["kind"] == "array_artifact"]
    expected_array_records = {
        (f"{outer}({inner})", f"{outer}_of_{inner}.pt")
        for outer, inner in config["pairs"]
    }
    observed_array_records = [
        (row.get("pair"), row.get("path")) for row in array_records
    ]
    if (
        len(observed_array_records) != len(expected_array_records)
        or set(observed_array_records) != expected_array_records
    ):
        failures.append("array artifact log identity mismatch")
    for row in array_records:
        path = array_dir / row["path"]
        if not path.is_file():
            failures.append(f"logged array artifact missing: {row['path']}")
        elif (
            sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["bytes"]
        ):
            failures.append(f"logged array artifact receipt mismatch: {row['path']}")
    return failures


def select_verdict(
    components_admitted: bool,
    failure_buckets: dict[str, list[str]],
) -> str:
    if not components_admitted:
        return "COMPONENT_ADMISSION_FAIL"
    if failure_buckets["integrity"]:
        return "INCOMPLETE"
    if failure_buckets["core"]:
        return "INSTRUMENT_FAIL"
    if failure_buckets["gauge"]:
        return "GAUGE_NUMERICAL_DRIFT"
    if failure_buckets["break"]:
        return "BREAK_NOT_OBSERVED"
    return "PASS_DICTIONARY_BREAK_REPAIR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = validate_config()
    registration_receipts = validate_execution_registration()

    torch.set_num_threads(config["threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    recorder = ChainedRecorder(run_dir / "results.jsonl")
    summaries: list[dict[str, Any]] = []
    assay_rows: list[dict[str, Any]] = []
    source_receipts: dict[str, str] = {}
    state_before: dict[str, str] = {}
    state_after: dict[str, str] = {}
    failure_buckets: dict[str, list[str]] = {
        "integrity": [],
        "core": [],
        "gauge": [],
        "break": [],
    }
    verdict = "INCOMPLETE"
    started = time.time()

    try:
        (run_dir / "components").mkdir()
        (run_dir / "arrays").mkdir()
        source_receipts = snapshot_sources(run_dir)
        validate_snapshot_against_registration(
            source_receipts, registration_receipts
        )
        environment = environment_record(config)
        write_json(run_dir / "environment.json", environment)
        recorder.log("run_start", config=config, environment=environment,
                     source_receipts=source_receipts,
                     registration_receipts=registration_receipts)

        components: dict[str, TinyTransformer] = {}
        for task in config["component_steps"]:
            model, summary = train_fixed_component(
                task, config, recorder, run_dir / "components"
            )
            summaries.append(summary)
            components[task] = model
        state_before = {
            task: state_dict_fingerprint(model)
            for task, model in components.items()
        }
        recorder.log("frozen_state_before", fingerprints=state_before)

        components_admitted = all(row["admitted"] for row in summaries)
        if not components_admitted:
            verdict = select_verdict(False, failure_buckets)
        else:
            failure_buckets["core"].extend(
                instrument_checks(components, recorder)
            )
            if failure_buckets["core"]:
                verdict = select_verdict(True, failure_buckets)
            else:
                for outer, inner in config["pairs"]:
                    rows, pair_buckets = assay_pair(
                        outer, inner, components, config, recorder,
                        run_dir / "arrays",
                    )
                    assay_rows.extend(rows)
                    for bucket, values in pair_buckets.items():
                        failure_buckets[bucket].extend(values)
                state_after = {
                    task: state_dict_fingerprint(model)
                    for task, model in components.items()
                }
                recorder.log("frozen_state_after", fingerprints=state_after)
                failure_buckets["integrity"].extend(reduce_integrity(
                    run_dir,
                    config,
                    summaries,
                    assay_rows,
                    source_receipts,
                    state_before,
                    state_after,
                ))
                verdict = select_verdict(True, failure_buckets)

        failures = [
            value
            for bucket in ("integrity", "core", "gauge", "break")
            for value in failure_buckets[bucket]
        ]
        summary_doc = {
            "components": summaries,
            "assay": assay_rows,
            "state_before": state_before,
            "state_after": state_after,
        }
        verdict_doc = {
            "verdict": verdict,
            "status": "PROVISIONAL_UNTIL_CHAIN_AND_MANIFEST_VERIFY",
            "failures": failures,
            "failure_buckets": failure_buckets,
            "component_count": len(summaries),
            "assay_cell_count": len(assay_rows),
            "scope": (
                "two selected, deterministically retrained E36-protocol "
                "seed-0 instruments; one previously observed derangement"
            ),
        }
        write_json(run_dir / "summary.json", summary_doc)
        write_json(run_dir / "verdict.json", verdict_doc)
        recorder.log("provisional_verdict", **verdict_doc)
        recorder.log("execution_complete", verdict=verdict,
                     elapsed_seconds=round(time.time() - started, 3))
    except BaseException as exc:
        verdict = "INCOMPLETE"
        error_doc = {
            "verdict": verdict,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(
            run_dir / "summary.json",
            {
                "components": summaries,
                "assay": assay_rows,
                "state_before": state_before,
                "state_after": state_after,
            },
        )
        write_json(
            run_dir / "verdict.json",
            {
                "verdict": verdict,
                "status": "INTERRUPTED",
                "failures": [repr(exc)],
                "failure_buckets": failure_buckets,
                "component_count": len(summaries),
                "assay_cell_count": len(assay_rows),
                "scope": "interrupted before registered completion",
            },
        )
        write_json(run_dir / "error.json", error_doc)
        recorder.log(
            "run_interrupted",
            **error_doc,
            component_count=len(summaries),
            assay_cell_count=len(assay_rows),
        )
        raise
    finally:
        recorder.close()
        chain = verify_result_chain(run_dir / "results.jsonl")
        write_json(run_dir / "result_chain.json", chain)
        manifest = make_manifest(run_dir)
        write_json(run_dir / "manifest.json", manifest)
        manifest_hash = sha256_file(run_dir / "manifest.json")
        write_text(
            run_dir / "manifest.sha256",
            manifest_hash + "  manifest.json\n", encoding="ascii",
        )

    completion = {
        "status": "FINAL",
        "verdict": verdict,
        "manifest_sha256": manifest_hash,
        "result_chain": chain,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(run_dir / "completion.json", completion)
    print(f"E57 complete: {verdict} in {(time.time() - started) / 60:.1f} min")
    print(f"manifest_sha256={manifest_hash}")
    print(f"completion_sha256={sha256_file(run_dir / 'completion.json')}")
    return 0 if verdict == "PASS_DICTIONARY_BREAK_REPAIR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
