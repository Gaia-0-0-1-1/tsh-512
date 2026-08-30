"""E58: prospective repair assay for E20's disconnected lattice router."""

from __future__ import annotations

import os

# These must precede Torch import/CUDA initialization. Conflicting inherited
# values fail immediately instead of being silently retained by setdefault.
_PREIMPORT_SETTINGS = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _name, _required in _PREIMPORT_SETTINGS.items():
    _observed = os.environ.get(_name)
    if _observed is not None and _observed != _required:
        raise RuntimeError(
            f"conflicting pre-import environment: {_name}={_observed!r}, "
            f"required {_required!r}"
        )
    os.environ[_name] = _required

import argparse
import copy
import ctypes
import hashlib
import json
import math
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
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "preregistered_config.json"
PREREG_PATH = HERE / "PREREG.md"
PREREGISTRATION_PATH = HERE / "preregistration.json"
REGISTRATION_PATH = HERE / "registration.json"
ATTEMPT_PATH = HERE / "attempt.json"
EXPECTED_CONFIG_SHA256 = (
    "18d517fa3f1456780e985616917038d702ced67721d523aebbcf5fba5a4078eb"
)
EXPECTED_GPU_UUID = "GPU-9d1cad1b-3dd7-5540-2522-a0e584a32555"
EXECUTION_SUBJECT = "e58/repaired-router-execution-registration"
LAYER_ORDER = ("q", "k", "v", "o", "w_in", "w_out", "unembed")
TASK_OFFSETS = {"Z2x2x2": 0, "Z8": 10000}
GUMBEL_CLAMP = (1.0e-6, 1.0 - 1.0e-6)
PHI = (1 + math.sqrt(5)) / 2
LATTICE_VALUES = {
    "ternary": (0.0, -1.0, 1.0),
    "phi1": (0.0, -1.0, 1.0, -PHI, PHI),
    "phi2": (
        0.0,
        -1.0,
        1.0,
        -1 / PHI,
        1 / PHI,
        -PHI,
        PHI,
        -PHI * PHI,
        PHI * PHI,
    ),
}

sys.path.insert(0, str(ROOT / "tools" / "e6"))
from math_structures import STRUCTURES  # noqa: E402


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


def mapping_fingerprint(mapping: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(mapping.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(canonical(list(value.shape)).encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


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


def atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    with temporary.open("r+b") as f:
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
        unsigned = {"seq": self.seq, "prev": self.prev, "kind": kind, **fields}
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


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self._f = path.open("x", encoding="utf-8", buffering=1)
        self.count = 0

    def write(self, value: Any) -> None:
        self._f.write(canonical(value) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self.count += 1

    def close(self) -> None:
        self._f.close()


def verify_result_chain(path: Path) -> dict[str, Any]:
    expected_prev = "0" * 64
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            digest = record.pop("hash")
            if record.get("seq") != count or record.get("prev") != expected_prev:
                raise RuntimeError(f"result-chain link mismatch at row {count}")
            observed = hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()
            if observed != digest:
                raise RuntimeError(f"result-chain hash mismatch at row {count}")
            expected_prev = digest
            count += 1
    return {"records": count, "head": expected_prev}


class WindowsExclusiveGpuLock:
    """Kernel-enforced no-share handle; the file itself is never deleted."""

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_ALWAYS = 4
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    def __init__(self, gpu_uuid: str):
        lock_dir = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        self.path = lock_dir / f"tsh512-{gpu_uuid}.lock"
        self.handle: int | None = None

    def acquire(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            str(self.path),
            self.GENERIC_READ | self.GENERIC_WRITE,
            0,
            None,
            self.OPEN_ALWAYS,
            self.FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if handle == self.INVALID_HANDLE_VALUE:
            error = ctypes.get_last_error()
            raise RuntimeError(f"GPU evidence lock unavailable (winerror={error})")
        self.handle = int(handle)

    def release(self) -> None:
        if self.handle is None:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        if not close_handle(ctypes.c_void_p(self.handle)):
            raise RuntimeError(
                f"failed to close GPU evidence lock (winerror={ctypes.get_last_error()})"
            )
        self.handle = None


class RoutedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str,
        bias: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.mode = mode
        for name, values in LATTICE_VALUES.items():
            self.register_buffer(
                f"table_{name}", torch.tensor(values, dtype=torch.float32)
            )
        if mode in {"dead", "live"}:
            self.router_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        else:
            self.register_parameter("router_logits", None)

    @staticmethod
    def _nearest(normalized: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        distances = (
            normalized.unsqueeze(-1)
            - table.reshape(*([1] * normalized.ndim), table.numel())
        ).abs()
        return distances.argmin(-1)

    def _quantized(self, lattice: str) -> torch.Tensor:
        latent = self.weight.detach()
        gamma = latent.abs().mean().clamp_min(1e-8)
        normalized = latent / gamma
        table = getattr(self, f"table_{lattice}")
        return table[self._nearest(normalized, table)] * gamma

    def _mixture(
        self,
        route_mode: str,
        gumbel: torch.Tensor | None,
        tau: float,
    ) -> torch.Tensor:
        if route_mode == "train":
            if gumbel is None or tuple(gumbel.shape) != (3,):
                raise RuntimeError("router training requires one explicit 3-vector")
            probabilities = torch.softmax((self.router_logits + gumbel) / tau, dim=0)
        elif route_mode == "soft":
            probabilities = torch.softmax(self.router_logits / tau, dim=0)
        elif route_mode == "hard":
            index = int(self.router_logits.argmax().item())
            probabilities = F.one_hot(
                torch.tensor(index, device=self.weight.device), num_classes=3
            ).to(self.weight.dtype)
        else:
            raise ValueError(f"unknown route mode {route_mode}")
        quantized = [self._quantized(name) for name in ("ternary", "phi1", "phi2")]
        return sum(probabilities[i] * quantized[i] for i in range(3))

    def forward(
        self,
        x: torch.Tensor,
        route_mode: str = "native",
        gumbel: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor:
        if self.mode == "fp":
            effective = self.weight
        elif self.mode in LATTICE_VALUES:
            quantized = self._quantized(self.mode)
            effective = self.weight + (quantized - self.weight).detach()
        elif self.mode in {"dead", "live"}:
            mixed = self._mixture(route_mode, gumbel, tau)
            dead = self.weight + (mixed - self.weight).detach()
            if self.mode == "dead":
                effective = dead
            else:
                # Do not recompute mixed. This zero-valued term restores only
                # the router derivative while preserving the exact forward.
                effective = dead + (mixed - mixed.detach())
        else:
            raise ValueError(f"unknown linear mode {self.mode}")
        return F.linear(x, effective, self.bias)


class RouterTransformer(nn.Module):
    def __init__(self, vocab: int, out_vocab: int, width: int, arm: str,
                 heterogeneous_assignment: dict[str, str], shadow_count: int):
        super().__init__()
        self.arm = arm
        self.width = width
        self.embed = nn.Embedding(vocab, width)
        self.pos = nn.Parameter(torch.zeros(1, 4, width))

        def mode_for(layer: str) -> str:
            if arm == "FP":
                return "fp"
            if arm == "PHI1":
                return "phi1"
            if arm == "HETERO_FIXED":
                return heterogeneous_assignment[layer]
            if arm == "LEGACY_DEAD":
                return "dead"
            if arm in {"LIVE_WD05", "LIVE_WD0"}:
                return "live"
            raise ValueError(f"unknown arm {arm}")

        self.q = RoutedLinear(width, width, mode_for("q"))
        self.k = RoutedLinear(width, width, mode_for("k"))
        self.v = RoutedLinear(width, width, mode_for("v"))
        self.o = RoutedLinear(width, width, mode_for("o"))
        self.ln1 = nn.LayerNorm(width)
        self.ln2 = nn.LayerNorm(width)
        self.w_in = RoutedLinear(width, 4 * width, mode_for("w_in"))
        self.w_out = RoutedLinear(4 * width, width, mode_for("w_out"))
        self.unembed = RoutedLinear(width, out_vocab, mode_for("unembed"))
        if arm in {"FP", "PHI1", "HETERO_FIXED"}:
            self.shadow = nn.Parameter(torch.zeros(shadow_count))
        else:
            self.register_parameter("shadow", None)

    def routed_layers(self) -> list[RoutedLinear]:
        return [getattr(self, name) for name in LAYER_ORDER]

    def forward(
        self,
        x: torch.Tensor,
        route_mode: str = "native",
        gumbel: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor:
        noises: list[torch.Tensor | None]
        if gumbel is None:
            noises = [None] * len(LAYER_ORDER)
        else:
            if tuple(gumbel.shape) != (len(LAYER_ORDER), 3):
                raise RuntimeError("Gumbel row must have shape [7,3]")
            noises = list(gumbel.unbind(0))

        e = self.embed(x) + self.pos[:, :x.shape[1]]
        z = self.ln1(e)
        q = self.q(z, route_mode, noises[0], tau)
        k = self.k(z, route_mode, noises[1], tau)
        v = self.v(z, route_mode, noises[2], tau)
        attention = (q @ k.transpose(-2, -1)) / (self.width ** 0.5)
        attention = attention.softmax(-1)
        h = e + self.o(attention @ v, route_mode, noises[3], tau)
        hidden = self.w_in(self.ln2(h), route_mode, noises[4], tau)
        h = h + self.w_out(F.relu(hidden), route_mode, noises[5], tau)
        return self.unembed(h.mean(dim=1), route_mode, noises[6], tau)

    def core_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name != "shadow" and not name.endswith("router_logits")
        ]

    def auxiliary_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name == "shadow" or name.endswith("router_logits")
        ]


def core_state(model: RouterTransformer) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.core_parameters()
    }


def load_core_state(model: RouterTransformer, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.core_parameters())
    if set(parameters) != set(state):
        raise RuntimeError("core parameter name mismatch")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def build_canonical_core(
    task: str, seed: int, config: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], str]:
    n = STRUCTURES[task]["n"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RouterTransformer(
            n, n, config["width"], "FP",
            config["heterogeneous_assignment"], config["shadow_parameters"],
        )
    state = core_state(model)
    return state, mapping_fingerprint(state)


def build_model(
    task: str,
    seed: int,
    arm: str,
    config: dict[str, Any],
    canonical_core: dict[str, torch.Tensor],
    device: torch.device,
) -> RouterTransformer:
    n = STRUCTURES[task]["n"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RouterTransformer(
            n, n, config["width"], arm,
            config["heterogeneous_assignment"], config["shadow_parameters"],
        )
    load_core_state(model, canonical_core)
    if mapping_fingerprint(core_state(model)) != mapping_fingerprint(canonical_core):
        raise RuntimeError("arm core initialization mismatch")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != config["nominal_parameters"]:
        raise RuntimeError(
            f"parameter mismatch for {arm}: {parameter_count} != "
            f"{config['nominal_parameters']}"
        )
    return model.to(device)


def canonical_domain(task: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = STRUCTURES[task]["n"]
    table = STRUCTURES[task]["make"]()
    x = torch.tensor([[a, b] for a in range(n) for b in range(n)], dtype=torch.long)
    y = torch.tensor([table[a][b] for a in range(n) for b in range(n)], dtype=torch.long)
    return x, y


def make_dataset(
    task: str, seed: int, config: dict[str, Any], shuffled: bool = False
) -> dict[str, Any]:
    x, y = canonical_domain(task)
    shuffle_permutation = None
    if shuffled:
        shuffle_generator = torch.Generator(device="cpu").manual_seed(
            config["shuffle_control"]["permutation_seed"]
        )
        shuffle_permutation = torch.randperm(len(y), generator=shuffle_generator)
        if torch.equal(shuffle_permutation, torch.arange(len(y))):
            raise RuntimeError("shuffle control permutation is identity")
        shuffled_y = y[shuffle_permutation]
        if not torch.equal(
            torch.bincount(shuffled_y, minlength=STRUCTURES[task]["n"]),
            torch.bincount(y, minlength=STRUCTURES[task]["n"]),
        ):
            raise RuntimeError("shuffle control changed label histogram")
        y = shuffled_y
    split_generator = torch.Generator(device="cpu").manual_seed(seed)
    split = torch.randperm(len(x), generator=split_generator)
    n_train = int(round(config["train_fraction"] * len(x)))
    train_indices, test_indices = split[:n_train], split[n_train:]
    return {
        "x": x,
        "y": y,
        "train_x": x[train_indices],
        "train_y": y[train_indices],
        "test_x": x[test_indices],
        "test_y": y[test_indices],
        "train_indices": train_indices,
        "test_indices": test_indices,
        "shuffle_permutation": shuffle_permutation,
        "fingerprints": {
            "x": tensor_fingerprint(x),
            "y": tensor_fingerprint(y),
            "train_indices": tensor_fingerprint(train_indices),
            "test_indices": tensor_fingerprint(test_indices),
            "train_x": tensor_fingerprint(x[train_indices]),
            "train_y": tensor_fingerprint(y[train_indices]),
            "test_x": tensor_fingerprint(x[test_indices]),
            "test_y": tensor_fingerprint(y[test_indices]),
            "shuffle_permutation": (
                tensor_fingerprint(shuffle_permutation)
                if shuffle_permutation is not None else None
            ),
        },
    }


def gumbel_schedule(
    task: str, seed: int, steps: int, config: dict[str, Any]
) -> tuple[torch.Tensor, int]:
    schedule_seed = config["gumbel_seed_offset"] + TASK_OFFSETS[task] + seed
    generator = torch.Generator(device="cpu").manual_seed(schedule_seed)
    uniform = torch.rand(
        (steps, len(LAYER_ORDER), 3), generator=generator, dtype=torch.float32
    ).clamp_(min=GUMBEL_CLAMP[0], max=GUMBEL_CLAMP[1])
    schedule = -torch.log(-torch.log(uniform))
    return schedule, schedule_seed


def optimizer_for(
    model: RouterTransformer, arm: str, config: dict[str, Any]
) -> torch.optim.Optimizer:
    core = [parameter for _, parameter in model.core_parameters()]
    auxiliary = [parameter for _, parameter in model.auxiliary_parameters()]
    if not auxiliary:
        raise RuntimeError(f"{arm} has no parity/router auxiliary parameters")
    auxiliary_wd = (
        config["router_weight_decay"][arm]
        if arm in config["router_weight_decay"] else config["core_weight_decay"]
    )
    return torch.optim.AdamW(
        [
            {"params": core, "weight_decay": config["core_weight_decay"]},
            {"params": auxiliary, "weight_decay": auxiliary_wd},
        ],
        lr=config["learning_rate"],
        betas=tuple(config["betas"]),
        foreach=False,
        fused=False,
    )


def router_parameters(model: RouterTransformer) -> list[nn.Parameter]:
    return [
        layer.router_logits
        for layer in model.routed_layers()
        if layer.router_logits is not None
    ]


def core_gradient_mapping(model: RouterTransformer) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.core_parameters()
    }


def router_gradient_status(model: RouterTransformer) -> dict[str, Any]:
    rows = []
    squared = 0.0
    for name, layer in zip(LAYER_ORDER, model.routed_layers()):
        parameter = layer.router_logits
        if parameter is None:
            continue
        gradient = parameter.grad
        present = gradient is not None
        finite = bool(present and torch.isfinite(gradient).all().item())
        norm = float(gradient.norm().item()) if present else None
        if norm is not None:
            squared += norm * norm
        rows.append({"layer": name, "present": present, "finite": finite, "norm": norm})
    return {"layers": rows, "aggregate_norm": math.sqrt(squared)}


def router_telemetry(
    model: RouterTransformer,
    initial_logits: list[torch.Tensor] | None,
    last_gradient: dict[str, Any] | None,
) -> dict[str, Any] | None:
    parameters = router_parameters(model)
    if not parameters:
        return None
    layers = []
    squared_delta = 0.0
    for index, (name, parameter) in enumerate(zip(LAYER_ORDER, parameters)):
        logits = parameter.detach().float().cpu()
        probabilities = torch.softmax(logits, dim=0)
        entropy = float(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
        )
        delta = (
            logits - initial_logits[index]
            if initial_logits is not None else torch.zeros_like(logits)
        )
        squared_delta += float(delta.square().sum().item())
        layers.append({
            "layer": name,
            "logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
            "entropy": entropy,
            "argmax": int(logits.argmax().item()),
            "centered_logit_norm": float((logits - logits.mean()).norm().item()),
        })
    choices = [row["argmax"] for row in layers]
    return {
        "layers": layers,
        "route_diversity": len(set(choices)),
        "cumulative_delta_norm": math.sqrt(squared_delta),
        "last_gradient": last_gradient,
    }


def parameter_norm(parameters: list[tuple[str, nn.Parameter]]) -> float:
    squared = sum(float(parameter.detach().square().sum().item()) for _, parameter in parameters)
    return math.sqrt(squared)


def evaluate_mode(
    model: RouterTransformer,
    dataset: dict[str, Any],
    device: torch.device,
    route_mode: str,
    tau: float,
) -> dict[str, float | int]:
    model.eval()
    train_x = dataset["train_x"].to(device)
    train_y = dataset["train_y"].to(device)
    test_x = dataset["test_x"].to(device)
    test_y = dataset["test_y"].to(device)
    with torch.no_grad():
        train_logits = model(train_x, route_mode=route_mode, tau=tau)
        test_logits = model(test_x, route_mode=route_mode, tau=tau)
        train_predictions = train_logits.argmax(-1)
        test_predictions = test_logits.argmax(-1)
        return {
            "train_loss": float(F.cross_entropy(train_logits, train_y).item()),
            "test_loss": float(F.cross_entropy(test_logits, test_y).item()),
            "train_accuracy": float((train_predictions == train_y).float().mean().item()),
            "test_accuracy": float((test_predictions == test_y).float().mean().item()),
            "train_correct": int((train_predictions == train_y).sum().item()),
            "test_correct": int((test_predictions == test_y).sum().item()),
        }


def evaluation_modes(arm: str) -> tuple[str, ...]:
    if arm in {"LEGACY_DEAD", "LIVE_WD05", "LIVE_WD0"}:
        return ("hard", "soft")
    return ("native",)


def primary_mode(arm: str) -> str:
    return "hard" if arm in {"LEGACY_DEAD", "LIVE_WD05", "LIVE_WD0"} else "native"


def evaluate_step(
    model: RouterTransformer,
    dataset: dict[str, Any],
    device: torch.device,
    arm: str,
    step: int,
    config: dict[str, Any],
    initial_logits: list[torch.Tensor] | None,
    last_gradient: dict[str, Any] | None,
) -> dict[str, Any]:
    modes = {
        mode: evaluate_mode(
            model, dataset, device, mode, config["gumbel_tau"]
        )
        for mode in evaluation_modes(arm)
    }
    return {
        "step": step,
        "modes": modes,
        "router": router_telemetry(model, initial_logits, last_gradient),
        "core_parameter_norm": parameter_norm(model.core_parameters()),
    }


def sustained_gate(
    curves: list[dict[str, Any]],
    mode: str,
    metric: str,
    threshold: float,
    sustain: int,
) -> dict[str, int | None]:
    qualifying = [row["modes"][mode][metric] >= threshold for row in curves]
    for end in range(sustain - 1, len(qualifying)):
        start = end - sustain + 1
        if all(qualifying[start:end + 1]):
            return {
                "onset_step": curves[start]["step"],
                "confirmation_step": curves[end]["step"],
            }
    return {"onset_step": None, "confirmation_step": None}


def summarize_curves(
    curves: list[dict[str, Any]], arm: str, config: dict[str, Any]
) -> dict[str, Any]:
    summaries = {}
    for mode in evaluation_modes(arm):
        memorization = sustained_gate(
            curves, mode, "train_accuracy", config["train_threshold"],
            config["sustain_evaluations"],
        )
        generalization = sustained_gate(
            curves, mode, "test_accuracy", config["test_threshold"],
            config["sustain_evaluations"],
        )
        mem_onset = memorization["onset_step"]
        gen_onset = generalization["onset_step"]
        if gen_onset is not None and (
            mem_onset is None or gen_onset < mem_onset
        ):
            classification = "GENERALIZED_WITHOUT_MEMORIZATION_GATE"
            delay = None if mem_onset is None else int(gen_onset - mem_onset)
        elif gen_onset is None:
            classification = "NO_SUSTAINED_GENERALIZATION"
            delay = None
        else:
            delay = int(gen_onset - mem_onset)
            classification = (
                "DELAYED_GROKKING"
                if delay >= config["min_grok_delay_steps"]
                else "ORDINARY_GENERALIZATION"
            )
        summaries[mode] = {
            "memorization": memorization,
            "generalization": generalization,
            "delay_steps": delay,
            "classification": classification,
            "terminal": curves[-1]["modes"][mode],
        }
    return summaries


def graph_gate(
    config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    task, seed = "Z2x2x2", config["seeds"][0]
    dataset = make_dataset(task, seed, config)
    canonical_core, core_fingerprint = build_canonical_core(task, seed, config)
    dead = build_model(task, seed, "LEGACY_DEAD", config, canonical_core, device)
    live = build_model(task, seed, "LIVE_WD05", config, canonical_core, device)
    schedule, schedule_seed = gumbel_schedule(task, seed, 1, config)
    noise = schedule[0].to(device)
    x = dataset["train_x"].to(device)
    y = dataset["train_y"].to(device)

    dead.train()
    live.train()
    dead_logits = dead(x, route_mode="train", gumbel=noise, tau=config["gumbel_tau"])
    live_logits = live(x, route_mode="train", gumbel=noise, tau=config["gumbel_tau"])
    forward_equal = torch.equal(dead_logits, live_logits)
    dead_loss = F.cross_entropy(dead_logits, y)
    live_loss = F.cross_entropy(live_logits, y)
    dead_loss.backward()
    live_loss.backward()
    dead_router = router_gradient_status(dead)
    live_router = router_gradient_status(live)
    dead_core_gradients = core_gradient_mapping(dead)
    live_core_gradients = core_gradient_mapping(live)
    core_gradient_equal = (
        set(dead_core_gradients) == set(live_core_gradients)
        and all(
            (dead_core_gradients[name] is None and live_core_gradients[name] is None)
            or (
                dead_core_gradients[name] is not None
                and live_core_gradients[name] is not None
                and torch.equal(dead_core_gradients[name], live_core_gradients[name])
            )
            for name in dead_core_gradients
        )
    )
    dead_before = [parameter.detach().clone() for parameter in router_parameters(dead)]
    live_before = [parameter.detach().clone() for parameter in router_parameters(live)]
    dead_optimizer = optimizer_for(dead, "LEGACY_DEAD", config)
    live_optimizer = optimizer_for(live, "LIVE_WD05", config)
    dead_optimizer.step()
    live_optimizer.step()
    dead_after = [parameter.detach().clone() for parameter in router_parameters(dead)]
    live_after = [parameter.detach().clone() for parameter in router_parameters(live)]
    dead_unchanged = all(torch.equal(a, b) for a, b in zip(dead_before, dead_after))
    live_changed = any(not torch.equal(a, b) for a, b in zip(live_before, live_after))
    core_after_equal = all(
        torch.equal(dict(dead.core_parameters())[name], dict(live.core_parameters())[name])
        for name in dict(dead.core_parameters())
    )
    dead_all_none = all(not row["present"] for row in dead_router["layers"])
    live_all_connected = all(
        row["present"] and row["finite"] for row in live_router["layers"]
    )
    passed = all((
        forward_equal,
        dead_all_none,
        live_all_connected,
        live_router["aggregate_norm"] > 0,
        dead_unchanged,
        live_changed,
        core_gradient_equal,
        core_after_equal,
    ))
    return {
        "passed": passed,
        "task": task,
        "seed": seed,
        "schedule_seed": schedule_seed,
        "schedule_fingerprint": tensor_fingerprint(schedule),
        "core_fingerprint": core_fingerprint,
        "forward_bit_identical": forward_equal,
        "dead_router_gradients": dead_router,
        "live_router_gradients": live_router,
        "dead_router_unchanged": dead_unchanged,
        "live_router_changed": live_changed,
        "core_gradients_bit_identical": core_gradient_equal,
        "post_step_core_bit_identical": core_after_equal,
        "dead_loss": float(dead_loss.item()),
        "live_loss": float(live_loss.item()),
    }


def deterministic_clone_gate(
    config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    task, seed, steps = "Z2x2x2", config["seeds"][0], 3
    dataset = make_dataset(task, seed, config)
    canonical_core, _ = build_canonical_core(task, seed, config)
    schedule, _ = gumbel_schedule(task, seed, steps, config)
    models = [
        build_model(task, seed, "LIVE_WD0", config, canonical_core, device)
        for _ in range(2)
    ]
    optimizers = [optimizer_for(model, "LIVE_WD0", config) for model in models]
    x = dataset["train_x"].to(device)
    y = dataset["train_y"].to(device)
    output_fingerprints = []
    gradient_fingerprints = []
    for step in range(steps):
        step_outputs = []
        step_gradients = []
        noise = schedule[step].to(device)
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, route_mode="train", gumbel=noise,
                           tau=config["gumbel_tau"])
            F.cross_entropy(logits, y).backward()
            gradients = torch.cat([
                parameter.grad.detach().flatten()
                for parameter in router_parameters(model)
            ])
            step_outputs.append(tensor_fingerprint(logits))
            step_gradients.append(tensor_fingerprint(gradients))
            optimizer.step()
        output_fingerprints.append(step_outputs)
        gradient_fingerprints.append(step_gradients)
    states = [mapping_fingerprint(model.state_dict()) for model in models]
    passed = all(pair[0] == pair[1] for pair in output_fingerprints) and all(
        pair[0] == pair[1] for pair in gradient_fingerprints
    ) and states[0] == states[1]
    return {
        "passed": passed,
        "output_fingerprints": output_fingerprints,
        "gradient_fingerprints": gradient_fingerprints,
        "terminal_state_fingerprints": states,
    }


def capability_score(
    summaries: list[dict[str, Any]], arm: str, mode: str, config: dict[str, Any]
) -> dict[str, Any]:
    rows = [
        row for row in summaries
        if not row["control"] and row["task"] == "Z8" and row["arm"] == arm
    ]
    confirmations = [
        row["gates"][mode]["generalization"]["confirmation_step"]
        for row in rows
    ]
    count = sum(value is not None for value in confirmations)
    restricted = [
        value if value is not None else config["z8_censor_step"]
        for value in confirmations
    ]
    memorization_count = sum(
        row["gates"][mode]["memorization"]["confirmation_step"] is not None
        for row in rows
    )
    return {
        "arm": arm,
        "mode": mode,
        "generalization_count": count,
        "restricted_mean_confirmation_step": sum(restricted) / len(restricted),
        "memorization_count": memorization_count,
        "confirmations": confirmations,
    }


def score_beats(a: dict[str, Any], b: dict[str, Any], config: dict[str, Any]) -> bool:
    if a["memorization_count"] < b["memorization_count"]:
        return False
    if a["generalization_count"] != b["generalization_count"]:
        return a["generalization_count"] > b["generalization_count"]
    return (
        a["restricted_mean_confirmation_step"]
        <= b["restricted_mean_confirmation_step"] - config["capability_margin_steps"]
    )


def score_predictions_and_verdict(
    summaries: list[dict[str, Any]],
    graph: dict[str, Any],
    clone_gate: dict[str, Any],
    integrity_failures: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    scores: dict[str, dict[str, Any]] = {}
    for arm in config["arms"]:
        for mode in evaluation_modes(arm):
            scores[f"{arm}/{mode}"] = capability_score(
                summaries, arm, mode, config
            )

    live_hard = scores["LIVE_WD0/hard"]
    live_soft = scores["LIVE_WD0/soft"]
    dead_hard = scores["LEGACY_DEAD/hard"]
    dead_soft = scores["LEGACY_DEAD/soft"]
    phi1 = scores["PHI1/native"]
    hetero = scores["HETERO_FIXED/native"]

    floor_counts = {
        arm: sum(
            row["gates"][primary_mode(arm)]["memorization"]["confirmation_step"]
            is not None
            for row in summaries
            if not row["control"] and row["task"] == "Z2x2x2" and row["arm"] == arm
        )
        for arm in config["arms"]
    }
    floor_pass = {arm: count >= 3 for arm, count in floor_counts.items()}
    shared_floor_failure = not any(floor_pass.values())

    control_generalization = [
        f"{row['cell_id']}/{mode}"
        for row in summaries if row["control"]
        for mode in evaluation_modes(row["arm"])
        if row["gates"][mode]["generalization"]["confirmation_step"]
        is not None
    ]
    p2_hit = floor_pass["LIVE_WD0"] and floor_pass["LEGACY_DEAD"] and score_beats(
        live_hard, dead_hard, config
    )
    hard_fixed_win = (
        floor_pass["LIVE_WD0"]
        and floor_pass["PHI1"]
        and floor_pass["HETERO_FIXED"]
        and score_beats(live_hard, phi1, config)
        and score_beats(live_hard, hetero, config)
    )
    soft_fixed_win = (
        floor_pass["LIVE_WD0"]
        and floor_pass["LEGACY_DEAD"]
        and floor_pass["PHI1"]
        and floor_pass["HETERO_FIXED"]
        and score_beats(live_soft, phi1, config)
        and score_beats(live_soft, hetero, config)
        and score_beats(live_soft, dead_soft, config)
    )
    successful_live = [
        row for row in summaries
        if not row["control"] and row["task"] == "Z8" and row["arm"] == "LIVE_WD0"
        and row["gates"]["hard"]["generalization"]["confirmation_step"] is not None
    ]
    diverse_success = any(
        row["primary_generalization_route_diversity"] is not None
        and row["primary_generalization_route_diversity"] >= 2
        for row in successful_live
    )
    hetero_only = (
        floor_pass["HETERO_FIXED"]
        and floor_pass["PHI1"]
        and floor_pass["LIVE_WD0"]
        and score_beats(hetero, phi1, config)
        and score_beats(hetero, live_hard, config)
        and score_beats(hetero, live_soft, config)
    )

    p1_hit = graph["passed"] and clone_gate["passed"]
    p3_hit = all(floor_pass.values())
    p4_hit = not control_generalization
    predictions = {
        "P1_graph_gate": {"hit": p1_hit},
        "P2_live_hard_beats_dead_hard": {"hit": p2_hit},
        "P3_all_arms_easy_floor": {
            "hit": p3_hit,
            "memorization_counts": floor_counts,
            "per_arm_pass": floor_pass,
        },
        "P4_no_shuffled_test_gate": {
            "hit": p4_hit,
            "violating_cells": control_generalization,
        },
    }

    if integrity_failures:
        verdict = "INCOMPLETE"
    elif not p1_hit:
        verdict = "INSTRUMENT_BROKEN"
    elif not p4_hit:
        verdict = "INVALID_LEAKAGE_CONTROL"
    elif shared_floor_failure:
        verdict = "OPTIMIZATION_FLOOR"
    elif p2_hit and hard_fixed_win and diverse_success:
        verdict = "ADAPTIVE_ROUTER_BENEFIT"
    elif p2_hit and hard_fixed_win:
        verdict = "GLOBAL_CHOICE_BENEFIT"
    elif soft_fixed_win and not (p2_hit and hard_fixed_win):
        verdict = "MIXTURE_ONLY_BENEFIT"
    elif hetero_only:
        verdict = "HETEROGENEOUS_REPRESENTATION_ONLY"
    else:
        verdict = "NO_BOUNDED_ROUTER_BENEFIT"
    return {
        "verdict": verdict,
        "predictions": predictions,
        "scores": scores,
        "mechanisms": {
            "p2_hit_but_top_verdict_may_differ": p2_hit,
            "hard_fixed_win": hard_fixed_win,
            "soft_fixed_win": soft_fixed_win,
            "diverse_success": diverse_success,
            "heterogeneous_only": hetero_only,
        },
        "integrity_failures": integrity_failures,
    }


def verify_registered_hashes(
    document_path: Path, expected_paths: set[str]
) -> dict[str, Any]:
    document = json.loads(document_path.read_text(encoding="utf-8"))
    hashes = document.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected_paths:
        raise RuntimeError(f"{document_path.name}: exact registered path set mismatch")
    receipts = {}
    root = ROOT.resolve()
    for relative, expected in sorted(hashes.items()):
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"registered path escapes root: {relative}") from error
        if not path.is_file():
            raise RuntimeError(f"registered path missing: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"registered hash mismatch for {relative}: {observed} != {expected}"
            )
        receipts[relative] = observed
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
            if record.get("seq") != expected_seq or record.get("prev") != expected_prev:
                raise RuntimeError(f"timeline link mismatch at row {row_index}")
            observed = hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()
            if observed != digest:
                raise RuntimeError(f"timeline hash mismatch at row {row_index}")
            records.append({**record, "hash": digest})
            expected_prev = digest
            expected_seq += 1
    return records


def validate_execution_registration() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("registered config digest mismatch")
    if not REGISTRATION_PATH.is_file():
        raise RuntimeError("execution registration is missing")
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if preregistration.get("base_commit") != head or registration.get("base_commit") != head:
        raise RuntimeError("registration base commit mismatch")
    prereg_paths = {
        "tools/e58/PREREG.md",
        "tools/e58/preregistered_config.json",
        "tools/e58/E20_ROUTER_AUDIT.md",
        "tools/e20/hyperbyte_test.py",
        "tools/e20/results.jsonl",
        "tools/e6/math_structures.py",
    }
    execution_paths = prereg_paths | {
        "tools/e58/preregistration.json",
        "tools/e58/router_repair.py",
        "tools/e58/test_router_repair.py",
    }
    prereg_receipt = verify_registered_hashes(PREREGISTRATION_PATH, prereg_paths)
    execution_receipt = verify_registered_hashes(REGISTRATION_PATH, execution_paths)
    preregistration_sha = sha256_file(PREREGISTRATION_PATH)
    if registration.get("preregistration_sha256") != preregistration_sha:
        raise RuntimeError("execution registration does not bind preregistration")
    if registration.get("execution_timeline_subject") != EXECUTION_SUBJECT:
        raise RuntimeError("unexpected execution timeline subject")

    records = timeline_records()
    prereg_anchor = registration.get("preregistration_timeline_anchor", {})
    prereg_matches = [
        record for record in records
        if record.get("seq") == prereg_anchor.get("seq")
    ]
    if len(prereg_matches) != 1:
        raise RuntimeError("invalid preregistration timeline anchor")
    anchored = prereg_matches[0]
    if (
        anchored.get("subject") != "e58/repaired-router-prereg"
        or anchored.get("kind") != "prediction"
        or anchored.get("payload_sha256") != preregistration_sha
    ):
        raise RuntimeError("preregistration anchor semantics mismatch")
    for field in ("hash", "subject", "payload_sha256"):
        if anchored.get(field) != prereg_anchor.get(field):
            raise RuntimeError(f"preregistration anchor mismatch for {field}")

    registration_sha = sha256_file(REGISTRATION_PATH)
    execution_matches = [
        record for record in records
        if record.get("subject") == EXECUTION_SUBJECT
        and record.get("payload_sha256") == registration_sha
    ]
    if len(execution_matches) != 1:
        raise RuntimeError("execution registration lacks a unique timeline anchor")
    execution_anchor = execution_matches[0]
    if (
        execution_anchor.get("kind") != "prediction"
        or execution_anchor.get("seq", -1) <= anchored.get("seq", -1)
    ):
        raise RuntimeError("execution anchor kind/order mismatch")
    return {
        "preregistration": prereg_receipt,
        "registration": execution_receipt,
        "preregistration_timeline_anchor": anchored,
        "execution_timeline_anchor": execution_anchor,
        "clarifications": registration.get("clarifications"),
    }


def validate_consumable_attempt(
    run_dir: Path, registration_receipts: dict[str, Any]
) -> dict[str, Any]:
    if not ATTEMPT_PATH.is_file():
        raise RuntimeError("single-use execution attempt is not registered")
    attempt = json.loads(ATTEMPT_PATH.read_text(encoding="utf-8"))
    expected_subject = "e58/repaired-router-attempt-1"
    if (
        attempt.get("attempt_number") != 1
        or Path(attempt.get("run_dir", "")).resolve() != run_dir.resolve()
        or attempt.get("registration_sha256")
        != registration_receipts["registration"]["document_sha256"]
        or attempt.get("timeline_subject") != expected_subject
        or attempt.get("interruption_policy")
        != "terminal INCOMPLETE; no retry or alternate run directory"
    ):
        raise RuntimeError("single-use execution attempt metadata mismatch")
    attempt_sha = sha256_file(ATTEMPT_PATH)
    records = timeline_records()
    attempt_records = [
        record for record in records
        if str(record.get("subject", "")).startswith(
            "e58/repaired-router-attempt-"
        )
    ]
    matching = [
        record for record in attempt_records
        if record.get("subject") == expected_subject
        and record.get("payload_sha256") == attempt_sha
    ]
    if len(attempt_records) != 1 or len(matching) != 1:
        raise RuntimeError("execution attempt is not uniquely timeline-anchored")
    anchor = matching[0]
    if (
        anchor.get("kind") != "prediction"
        or anchor.get("seq", -1)
        <= registration_receipts["execution_timeline_anchor"].get("seq", -1)
    ):
        raise RuntimeError("execution attempt anchor kind/order mismatch")
    return {
        "document": ATTEMPT_PATH.name,
        "document_sha256": attempt_sha,
        "timeline_anchor": anchor,
        "policy": attempt["interruption_policy"],
    }


def snapshot_sources(run_dir: Path) -> dict[str, str]:
    source_dir = run_dir / "source"
    source_dir.mkdir()
    sources = {
        "PREREG.md": PREREG_PATH,
        "preregistered_config.json": CONFIG_PATH,
        "E20_ROUTER_AUDIT.md": HERE / "E20_ROUTER_AUDIT.md",
        "preregistration.json": PREREGISTRATION_PATH,
        "registration.json": REGISTRATION_PATH,
        "attempt.json": ATTEMPT_PATH,
        "router_repair.py": Path(__file__),
        "test_router_repair.py": HERE / "test_router_repair.py",
        "hyperbyte_test.py": ROOT / "tools" / "e20" / "hyperbyte_test.py",
        "e20_results.jsonl": ROOT / "tools" / "e20" / "results.jsonl",
        "math_structures.py": ROOT / "tools" / "e6" / "math_structures.py",
    }
    receipts = {}
    for name, source in sources.items():
        target = source_dir / name
        shutil.copy2(source, target)
        receipts[name] = sha256_file(target)
    return receipts


def validate_snapshot_receipts(
    receipts: dict[str, str], registration_receipts: dict[str, Any]
) -> None:
    rename = {"results.jsonl": "e20_results.jsonl"}
    expected = {
        rename.get(Path(relative).name, Path(relative).name): digest
        for relative, digest in registration_receipts["registration"]["files"].items()
    }
    expected["registration.json"] = registration_receipts["registration"][
        "document_sha256"
    ]
    expected["attempt.json"] = registration_receipts["attempt"][
        "document_sha256"
    ]
    if receipts != expected:
        raise RuntimeError("source snapshots differ from execution registration")


def nvidia_sample() -> dict[str, Any]:
    fields = [
        "uuid", "pci.bus_id", "name", "driver_version", "pstate",
        "utilization.gpu", "memory.used", "memory.total",
    ]
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(f"expected one visible GPU, observed {len(output)}")
    values = [value.strip() for value in output[0].split(",")]
    row = dict(zip(fields, values))
    row["utilization.gpu"] = int(row["utilization.gpu"])
    row["memory.used"] = int(row["memory.used"])
    row["memory.total"] = int(row["memory.total"])
    row["sampled_at"] = time.time()
    return row


def python_process_snapshot() -> list[dict[str, Any]]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object {$_.Name -match 'python'} | "
        "Select-Object ProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Compress"
    )
    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not output:
        return []
    parsed = json.loads(output)
    return parsed if isinstance(parsed, list) else [parsed]


def nvidia_compute_process_snapshot() -> list[dict[str, Any]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not output:
        return []
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) != 3:
            raise RuntimeError(f"unexpected nvidia compute row: {line}")
        pid_raw, name, memory_raw = values
        rows.append({
            "pid": int(pid_raw) if pid_raw.isdigit() else pid_raw,
            "process_name": name,
            "used_memory": (
                int(memory_raw) if memory_raw.isdigit() else memory_raw
            ),
        })
    return rows


def configure_and_preflight_cuda(config: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG was not fixed before import")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "PYTHONHASHSEED=0 must be set by the parent before interpreter start"
        )
    torch.set_num_threads(config["threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]

    samples = []
    for index in range(10):
        samples.append(nvidia_sample())
        if index != 9:
            time.sleep(0.5)
    utilizations = [row["utilization.gpu"] for row in samples]
    memories = [row["memory.used"] for row in samples]
    sorted_utilization = sorted(utilizations)
    median_utilization = (
        sorted_utilization[4] + sorted_utilization[5]
    ) / 2
    if (
        any(row["uuid"] != EXPECTED_GPU_UUID for row in samples)
        or median_utilization > 10
        or max(utilizations) > 20
        or max(memories) > 2048
        or max(memories) - min(memories) > 256
    ):
        raise RuntimeError("GPU idle/identity preflight failed")

    processes = python_process_snapshot()
    compute_processes = nvidia_compute_process_snapshot()
    foreign_evidence = []
    for process in processes:
        pid = process.get("ProcessId")
        command = (process.get("CommandLine") or "").lower().replace("/", "\\")
        if pid == os.getpid():
            continue
        if any(token in command for token in (
            "tools\\e56", "tools\\e57", "tools\\e58", "router_repair.py"
        )):
            foreign_evidence.append(process)
    if foreign_evidence:
        raise RuntimeError(f"another evidence process is active: {foreign_evidence}")
    python_pids = {
        int(row["ProcessId"])
        for row in processes
        if row.get("ProcessId") is not None
    }
    foreign_python_compute = [
        row for row in compute_processes
        if row.get("pid") != os.getpid()
        and (
            row.get("pid") in python_pids
            or "python" in str(row.get("process_name", "")).lower()
        )
    ]
    if foreign_python_compute:
        raise RuntimeError(
            f"another Python CUDA context is active: {foreign_python_compute}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    if tuple((properties.major, properties.minor)) != (12, 0):
        raise RuntimeError("unexpected CUDA compute capability")
    flags = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    expected_flags = {
        "deterministic_algorithms": True,
        "threads": 1,
        "interop_threads": 1,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "float32_matmul_precision": "highest",
    }
    for key, expected in expected_flags.items():
        if flags[key] != expected:
            raise RuntimeError(f"deterministic preflight mismatch: {key}")
    return {
        "samples": samples,
        "median_utilization": median_utilization,
        "python_processes": processes,
        "nvidia_compute_processes": compute_processes,
        "flags": flags,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": {
            "name": properties.name,
            "major": properties.major,
            "minor": properties.minor,
            "total_memory": properties.total_memory,
        },
    }


def cell_id(task: str, arm: str, seed: int, control: bool) -> str:
    prefix = "control" if control else "main"
    return f"{prefix}__{task}__{arm}__s{seed}"


def run_cell(
    task: str,
    arm: str,
    seed: int,
    control: bool,
    steps: int,
    dataset: dict[str, Any],
    schedule: torch.Tensor,
    schedule_fingerprint: str,
    canonical_core: dict[str, torch.Tensor],
    core_fingerprint: str,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    identifier = cell_id(task, arm, seed, control)
    model = build_model(
        task, seed, arm, config, canonical_core, device
    )
    if mapping_fingerprint(core_state(model)) != core_fingerprint:
        raise RuntimeError(f"{identifier}: core state mismatch before training")
    optimizer = optimizer_for(model, arm, config)
    initial_logits = [
        parameter.detach().cpu().clone() for parameter in router_parameters(model)
    ] or None
    route_schedule = (
        schedule.to(device)
        if arm in {"LEGACY_DEAD", "LIVE_WD05", "LIVE_WD0"} else None
    )
    x_train = dataset["train_x"].to(device)
    y_train = dataset["train_y"].to(device)
    curve_path = run_dir / "curves" / f"{identifier}.jsonl"
    curve_writer = JsonlWriter(curve_path)
    curves: list[dict[str, Any]] = []
    last_gradient = None
    evaluation_receipts = {
        "core_initial_fingerprint": core_fingerprint,
        "dataset_fingerprints": dataset["fingerprints"],
        "schedule_fingerprint": schedule_fingerprint,
    }
    started = time.time()
    try:
        row = evaluate_step(
            model, dataset, device, arm, 0, config, initial_logits,
            last_gradient,
        )
        row["receipts"] = evaluation_receipts
        curve_writer.write(row)
        curves.append(row)
        for step in range(1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if route_schedule is None:
                logits = model(x_train, route_mode="native")
            else:
                logits = model(
                    x_train,
                    route_mode="train",
                    gumbel=route_schedule[step - 1],
                    tau=config["gumbel_tau"],
                )
            loss = F.cross_entropy(logits, y_train)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"{identifier}: nonfinite training loss at {step}")
            loss.backward()
            if route_schedule is not None:
                last_gradient = router_gradient_status(model)
                if arm == "LEGACY_DEAD":
                    if any(row_["present"] for row_ in last_gradient["layers"]):
                        raise RuntimeError(f"{identifier}: dead router acquired gradient")
                elif (
                    not all(
                        row_["present"] and row_["finite"]
                        for row_ in last_gradient["layers"]
                    )
                ):
                    raise RuntimeError(f"{identifier}: live router gradient failed")
            optimizer.step()
            if step % config["eval_every"] == 0:
                row = evaluate_step(
                    model, dataset, device, arm, step, config,
                    initial_logits, last_gradient,
                )
                row["receipts"] = evaluation_receipts
                curve_writer.write(row)
                curves.append(row)
    finally:
        curve_writer.close()

    expected_evaluations = steps // config["eval_every"] + 1
    if len(curves) != expected_evaluations:
        raise RuntimeError(f"{identifier}: evaluation count mismatch")
    gates = summarize_curves(curves, arm, config)
    primary = primary_mode(arm)
    terminal_router = curves[-1]["router"]
    primary_confirmation = gates[primary]["generalization"][
        "confirmation_step"
    ]
    confirmation_row = next(
        (row for row in curves if row["step"] == primary_confirmation),
        None,
    )
    primary_generalization_route_diversity = None
    if confirmation_row is not None:
        primary_generalization_route_diversity = (
            confirmation_row["router"]["route_diversity"]
            if confirmation_row["router"] is not None else 1
        )
    checkpoint_path = run_dir / "checkpoints" / f"{identifier}.pt"
    checkpoint = {
        "cell_id": identifier,
        "task": task,
        "arm": arm,
        "seed": seed,
        "control": control,
        "step": steps,
        "model": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "core_initial_fingerprint": core_fingerprint,
        "dataset_fingerprints": dataset["fingerprints"],
        "schedule_fingerprint": schedule_fingerprint,
        "schedule_index": steps,
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    terminal_state_fingerprint = mapping_fingerprint(checkpoint["model"])
    summary = {
        "cell_id": identifier,
        "task": task,
        "arm": arm,
        "seed": seed,
        "control": control,
        "steps": steps,
        "evaluations": len(curves),
        "primary_mode": primary,
        "gates": gates,
        "terminal_route_diversity": (
            terminal_router["route_diversity"] if terminal_router is not None else 1
        ),
        "primary_generalization_route_diversity": (
            primary_generalization_route_diversity
        ),
        "terminal_router": terminal_router,
        "core_initial_fingerprint": core_fingerprint,
        "terminal_state_fingerprint": terminal_state_fingerprint,
        "dataset_fingerprints": dataset["fingerprints"],
        "schedule_fingerprint": schedule_fingerprint,
        "curve": curve_path.name,
        "curve_sha256": sha256_file(curve_path),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "seconds": round(time.time() - started, 3),
    }
    summary_path = run_dir / "cells" / f"{identifier}.json"
    write_json(summary_path, summary)
    summary["summary"] = summary_path.name
    summary["summary_sha256"] = sha256_file(summary_path)
    print(canonical({
        "cell": identifier,
        "classification": gates[primary]["classification"],
        "mem": gates[primary]["memorization"]["confirmation_step"],
        "gen": gates[primary]["generalization"]["confirmation_step"],
        "seconds": summary["seconds"],
    }), flush=True)
    return summary


def expected_cell_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {}
    for task, steps in config["tasks"].items():
        for seed in config["seeds"]:
            for arm in config["arms"]:
                identifier = cell_id(task, arm, seed, False)
                specs[identifier] = {
                    "task": task, "arm": arm, "seed": seed,
                    "control": False, "steps": steps,
                }
    control_task = config["shuffle_control"]["task"]
    control_steps = config["tasks"][control_task]
    for seed in config["seeds"]:
        for arm in config["shuffle_control"]["arms"]:
            identifier = cell_id(control_task, arm, seed, True)
            specs[identifier] = {
                "task": control_task, "arm": arm, "seed": seed,
                "control": True, "steps": control_steps,
            }
    return specs


def reduce_integrity(
    run_dir: Path,
    summaries: list[dict[str, Any]],
    schedules: dict[tuple[str, int], dict[str, Any]],
    source_receipts: dict[str, str],
    registration_receipts: dict[str, Any],
    graph: dict[str, Any],
    clone_gate: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected = expected_cell_specs(config)
    observed_ids = [row["cell_id"] for row in summaries]
    if len(observed_ids) != len(expected) or set(observed_ids) != set(expected):
        failures.append("terminal cell identity mismatch")
        return failures
    if len(expected) != 56:
        failures.append("registered expected-cell reducer is not 56")
    validate_snapshot_receipts(source_receipts, registration_receipts)
    source_dir = run_dir / "source"
    observed_sources = {
        path.name for path in source_dir.iterdir() if path.is_file()
    }
    if observed_sources != set(source_receipts):
        failures.append("source snapshot file set mismatch")
    for name in sorted(observed_sources & set(source_receipts)):
        if sha256_file(source_dir / name) != source_receipts[name]:
            failures.append(f"source snapshot hash mismatch: {name}")
    if json.loads((run_dir / "graph_gate.json").read_text(encoding="utf-8")) != graph:
        failures.append("graph-gate artifact mismatch")
    if json.loads(
        (run_dir / "deterministic_clone_gate.json").read_text(encoding="utf-8")
    ) != clone_gate:
        failures.append("deterministic-clone artifact mismatch")

    expected_dataset_files = {
        f"main__{task}__s{seed}.pt"
        for task in config["tasks"] for seed in config["seeds"]
    } | {
        f"control__{config['shuffle_control']['task']}__s{seed}.pt"
        for seed in config["seeds"]
    }
    observed_dataset_files = {
        path.name for path in (run_dir / "datasets").iterdir() if path.is_file()
    }
    if observed_dataset_files != expected_dataset_files:
        failures.append("dataset artifact set mismatch")
    datasets: dict[tuple[str, int, bool], dict[str, Any]] = {}
    for name in sorted(expected_dataset_files & observed_dataset_files):
        parts = name.removesuffix(".pt").split("__")
        control = parts[0] == "control"
        task = parts[1]
        seed = int(parts[2][1:])
        dataset = torch.load(
            run_dir / "datasets" / name, map_location="cpu", weights_only=True
        )
        recomputed = {
            "x": tensor_fingerprint(dataset["x"]),
            "y": tensor_fingerprint(dataset["y"]),
            "train_indices": tensor_fingerprint(dataset["train_indices"]),
            "test_indices": tensor_fingerprint(dataset["test_indices"]),
            "train_x": tensor_fingerprint(dataset["train_x"]),
            "train_y": tensor_fingerprint(dataset["train_y"]),
            "test_x": tensor_fingerprint(dataset["test_x"]),
            "test_y": tensor_fingerprint(dataset["test_y"]),
            "shuffle_permutation": (
                tensor_fingerprint(dataset["shuffle_permutation"])
                if dataset["shuffle_permutation"] is not None else None
            ),
        }
        if recomputed != dataset["fingerprints"]:
            failures.append(f"dataset fingerprint mismatch: {name}")
        fresh = make_dataset(task, seed, config, shuffled=control)
        if recomputed != fresh["fingerprints"]:
            failures.append(f"dataset replay mismatch: {name}")
        datasets[(task, seed, control)] = dataset

    expected_initializations = {
        f"{task}__s{seed}.pt"
        for task in config["tasks"] for seed in config["seeds"]
    }
    observed_initializations = {
        path.name
        for path in (run_dir / "initializations").iterdir()
        if path.is_file()
    }
    if observed_initializations != expected_initializations:
        failures.append("initialization artifact set mismatch")
    initial_fingerprints: dict[tuple[str, int], str] = {}
    initial_states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for name in sorted(expected_initializations & observed_initializations):
        artifact = torch.load(
            run_dir / "initializations" / name,
            map_location="cpu",
            weights_only=True,
        )
        fingerprint = mapping_fingerprint(artifact["core"])
        if fingerprint != artifact["fingerprint"]:
            failures.append(f"initialization fingerprint mismatch: {name}")
        fresh_core, fresh_fingerprint = build_canonical_core(
            artifact["task"], artifact["seed"], config
        )
        if (
            fingerprint != fresh_fingerprint
            or mapping_fingerprint(fresh_core) != fingerprint
        ):
            failures.append(f"initialization replay mismatch: {name}")
        initial_fingerprints[(artifact["task"], artifact["seed"])] = fingerprint
        initial_states[(artifact["task"], artifact["seed"])] = artifact["core"]

    summary_by_id = {row["cell_id"]: row for row in summaries}
    expected_file_names = {f"{identifier}.json" for identifier in expected}
    observed_summary_files = {
        path.name for path in (run_dir / "cells").iterdir() if path.is_file()
    }
    if observed_summary_files != expected_file_names:
        failures.append("cell-summary file set mismatch")
    observed_curve_files = {
        path.name for path in (run_dir / "curves").iterdir() if path.is_file()
    }
    if observed_curve_files != {f"{identifier}.jsonl" for identifier in expected}:
        failures.append("curve file set mismatch")
    observed_checkpoints = {
        path.name for path in (run_dir / "checkpoints").iterdir() if path.is_file()
    }
    if observed_checkpoints != {f"{identifier}.pt" for identifier in expected}:
        failures.append("checkpoint file set mismatch")

    for identifier, spec in expected.items():
        summary = summary_by_id[identifier]
        if any(summary[key] != spec[key] for key in spec):
            failures.append(f"cell metadata mismatch: {identifier}")
        curve_path = run_dir / "curves" / f"{identifier}.jsonl"
        checkpoint_path = run_dir / "checkpoints" / f"{identifier}.pt"
        summary_path = run_dir / "cells" / f"{identifier}.json"
        if sha256_file(curve_path) != summary["curve_sha256"]:
            failures.append(f"curve hash mismatch: {identifier}")
        if sha256_file(checkpoint_path) != summary["checkpoint_sha256"]:
            failures.append(f"checkpoint hash mismatch: {identifier}")
        if sha256_file(summary_path) != summary["summary_sha256"]:
            failures.append(f"summary hash mismatch: {identifier}")
        summary_artifact = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_without_receipt = {
            key: value for key, value in summary.items()
            if key not in {"summary", "summary_sha256"}
        }
        if canonical(summary_artifact) != canonical(summary_without_receipt):
            failures.append(f"summary content mismatch: {identifier}")
        with curve_path.open("r", encoding="utf-8") as f:
            curves = [json.loads(line) for line in f]
        expected_steps = list(range(0, spec["steps"] + 1, config["eval_every"]))
        if [row["step"] for row in curves] != expected_steps:
            failures.append(f"curve step schedule mismatch: {identifier}")
        if len(curves) != summary["evaluations"]:
            failures.append(f"curve evaluation count mismatch: {identifier}")
        if any(set(row["modes"]) != set(evaluation_modes(spec["arm"])) for row in curves):
            failures.append(f"curve mode set mismatch: {identifier}")
        expected_evaluation_receipts = {
            "core_initial_fingerprint": summary["core_initial_fingerprint"],
            "dataset_fingerprints": summary["dataset_fingerprints"],
            "schedule_fingerprint": summary["schedule_fingerprint"],
        }
        if any(
            row.get("receipts") != expected_evaluation_receipts
            for row in curves
        ):
            failures.append(f"curve receipt binding mismatch: {identifier}")
        for row in curves:
            for metrics in row["modes"].values():
                if (
                    abs(
                        metrics["train_accuracy"]
                        - metrics["train_correct"] / 51
                    ) > 1e-7
                    or abs(
                        metrics["test_accuracy"]
                        - metrics["test_correct"] / 13
                    ) > 1e-7
                ):
                    failures.append(f"curve accuracy arithmetic mismatch: {identifier}")
                    break
        recomputed_gates = summarize_curves(curves, spec["arm"], config)
        if canonical(recomputed_gates) != canonical(summary["gates"]):
            failures.append(f"gate reduction mismatch: {identifier}")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            checkpoint["cell_id"] != identifier
            or checkpoint["task"] != spec["task"]
            or checkpoint["arm"] != spec["arm"]
            or checkpoint["seed"] != spec["seed"]
            or checkpoint["control"] != spec["control"]
            or checkpoint["step"] != spec["steps"]
        ):
            failures.append(f"checkpoint metadata mismatch: {identifier}")
        if mapping_fingerprint(checkpoint["model"]) != summary[
            "terminal_state_fingerprint"
        ]:
            failures.append(f"checkpoint state mismatch: {identifier}")
        expected_dataset = datasets[(spec["task"], spec["seed"], spec["control"])]
        if (
            summary["dataset_fingerprints"] != expected_dataset["fingerprints"]
            or checkpoint["dataset_fingerprints"] != expected_dataset["fingerprints"]
        ):
            failures.append(f"dataset binding mismatch: {identifier}")
        init_fingerprint = initial_fingerprints[(spec["task"], spec["seed"])]
        if (
            summary["core_initial_fingerprint"] != init_fingerprint
            or checkpoint["core_initial_fingerprint"] != init_fingerprint
        ):
            failures.append(f"initialization binding mismatch: {identifier}")
        schedule_receipt = schedules[(spec["task"], spec["seed"])]
        if (
            summary["schedule_fingerprint"] != schedule_receipt["fingerprint"]
            or checkpoint["schedule_fingerprint"] != schedule_receipt["fingerprint"]
            or checkpoint["schedule_index"] != spec["steps"]
        ):
            failures.append(f"schedule binding mismatch: {identifier}")
        expected_router = curves[-1]["router"]
        if canonical(summary["terminal_router"]) != canonical(expected_router):
            failures.append(f"terminal router telemetry mismatch: {identifier}")
        expected_terminal_diversity = (
            expected_router["route_diversity"]
            if expected_router is not None else 1
        )
        if summary["terminal_route_diversity"] != expected_terminal_diversity:
            failures.append(f"terminal route diversity mismatch: {identifier}")
        confirmation_step = summary["gates"][summary["primary_mode"]][
            "generalization"
        ]["confirmation_step"]
        confirmation_row = next(
            (row for row in curves if row["step"] == confirmation_step),
            None,
        )
        expected_confirmation_diversity = None
        if confirmation_row is not None:
            expected_confirmation_diversity = (
                confirmation_row["router"]["route_diversity"]
                if confirmation_row["router"] is not None else 1
            )
        if (
            summary["primary_generalization_route_diversity"]
            != expected_confirmation_diversity
        ):
            failures.append(
                f"generalization route diversity mismatch: {identifier}"
            )

        replay_model = build_model(
            spec["task"], spec["seed"], spec["arm"], config,
            initial_states[(spec["task"], spec["seed"])],
            torch.device(config["device"]),
        )
        replay_model.load_state_dict(checkpoint["model"], strict=True)
        replay_modes = {
            mode: evaluate_mode(
                replay_model,
                expected_dataset,
                torch.device(config["device"]),
                mode,
                config["gumbel_tau"],
            )
            for mode in evaluation_modes(spec["arm"])
        }
        if canonical(replay_modes) != canonical(curves[-1]["modes"]):
            failures.append(f"checkpoint terminal metric replay mismatch: {identifier}")
        replay_initial_logits = [
            torch.zeros_like(parameter.detach().cpu())
            for parameter in router_parameters(replay_model)
        ] or None
        replay_router = router_telemetry(
            replay_model,
            replay_initial_logits,
            expected_router["last_gradient"] if expected_router is not None else None,
        )
        if canonical(replay_router) != canonical(expected_router):
            failures.append(f"checkpoint router telemetry replay mismatch: {identifier}")
        del replay_model

    expected_schedule_keys = {
        (task, seed) for task in config["tasks"] for seed in config["seeds"]
    }
    if set(schedules) != expected_schedule_keys:
        failures.append("schedule key set mismatch")
    expected_schedule_files = {
        f"{task}__s{seed}.pt" for task, seed in expected_schedule_keys
    }
    observed_schedule_files = {
        path.name for path in (run_dir / "schedules").iterdir()
        if path.is_file()
    }
    if observed_schedule_files != expected_schedule_files:
        failures.append("schedule artifact file set mismatch")
    for key, receipt in schedules.items():
        path = run_dir / "schedules" / receipt["path"]
        task, seed = key
        expected_path = f"{task}__s{seed}.pt"
        expected_steps = config["tasks"][task]
        expected_seed = config["gumbel_seed_offset"] + TASK_OFFSETS[task] + seed
        if (
            receipt["task"] != task
            or receipt["seed"] != seed
            or receipt["path"] != expected_path
            or receipt["schedule_seed"] != expected_seed
            or receipt["seed_formula"] != expected_seed
            or receipt["shape"] != [expected_steps, len(LAYER_ORDER), 3]
            or receipt["dtype"] != "torch.float32"
        ):
            failures.append(f"schedule receipt metadata mismatch: {key}")
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            failures.append(f"schedule artifact mismatch: {key}")
        else:
            schedule = torch.load(path, map_location="cpu", weights_only=True)
            replay, replay_seed = gumbel_schedule(
                task, seed, expected_steps, config
            )
            if (
                tensor_fingerprint(schedule) != receipt["fingerprint"]
                or list(schedule.shape) != receipt["shape"]
                or str(schedule.dtype) != receipt["dtype"]
                or replay_seed != expected_seed
                or not torch.equal(schedule, replay)
                or tensor_fingerprint(replay) != receipt["fingerprint"]
            ):
                failures.append(f"schedule tensor mismatch: {key}")
    verify_result_chain(run_dir / "results.jsonl")
    with (run_dir / "results.jsonl").open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    terminal_records = []
    for record in records:
        if record["kind"] == "cell_terminal":
            terminal_records.append({
                key: value for key, value in record.items()
                if key not in {"seq", "prev", "hash", "kind"}
            })
    if (
        len(terminal_records) != len(summaries)
        or [canonical(row) for row in terminal_records]
        != [canonical(row) for row in summaries]
    ):
        failures.append("cell terminal result-chain mismatch")
    return failures


def make_manifest(run_dir: Path) -> dict[str, Any]:
    excluded = {
        "manifest.json", "manifest.sha256", "completion.json",
        "gpu_lock_release.json", "seal.json",
    }
    files = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name in excluded:
            continue
        files.append({
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {"files": files}


def environment_record(
    config: dict[str, Any], preflight: dict[str, Any]
) -> dict[str, Any]:
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
        "preflight": preflight,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "git_status": git_status,
        "timeline_sha256": sha256_file(ROOT / "timeline.jsonl"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "environment": {
            key: os.environ.get(key) for key in (
                "CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_ORDER",
                "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    registration_receipts = validate_execution_registration()

    run_dir = Path(args.run_dir).resolve()
    registration_receipts["attempt"] = validate_consumable_attempt(
        run_dir, registration_receipts
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "cells", "curves", "checkpoints", "schedules", "datasets",
        "initializations",
    ):
        (run_dir / name).mkdir()
    recorder = ChainedRecorder(run_dir / "results.jsonl")
    lock_events = JsonlWriter(run_dir / "gpu_lock_events.jsonl")
    gpu_lock = WindowsExclusiveGpuLock(EXPECTED_GPU_UUID)
    lock_acquired = False
    summaries: list[dict[str, Any]] = []
    schedules: dict[tuple[str, int], dict[str, Any]] = {}
    source_receipts: dict[str, str] = {}
    verdict = "INCOMPLETE"
    result_doc: dict[str, Any] = {}
    started = time.time()

    try:
        gpu_lock.acquire()
        lock_acquired = True
        lock_events.write({
            "kind": "acquire",
            "pid": os.getpid(),
            "time": time.time(),
            "gpu_uuid": EXPECTED_GPU_UUID,
            "lock_path": str(gpu_lock.path),
        })
        source_receipts = snapshot_sources(run_dir)
        validate_snapshot_receipts(source_receipts, registration_receipts)
        preflight = configure_and_preflight_cuda(config)
        environment = environment_record(config, preflight)
        write_json(run_dir / "environment.json", environment)
        lock_events.write({
            "kind": "preflight_pass",
            "time": time.time(),
            "median_utilization": preflight["median_utilization"],
            "last_sample": preflight["samples"][-1],
        })
        recorder.log(
            "run_start",
            config=config,
            environment=environment,
            source_receipts=source_receipts,
            registration_receipts=registration_receipts,
        )

        device = torch.device(config["device"])
        graph = graph_gate(config, device)
        clone_gate = deterministic_clone_gate(config, device)
        write_json(run_dir / "graph_gate.json", graph)
        write_json(run_dir / "deterministic_clone_gate.json", clone_gate)
        recorder.log("graph_gate", **graph)
        recorder.log("deterministic_clone_gate", **clone_gate)

        if graph["passed"] and clone_gate["passed"]:
            for task, steps in config["tasks"].items():
                for seed in config["seeds"]:
                    schedule, schedule_seed = gumbel_schedule(
                        task, seed, steps, config
                    )
                    schedule_name = f"{task}__s{seed}.pt"
                    schedule_path = run_dir / "schedules" / schedule_name
                    atomic_torch_save(schedule_path, schedule)
                    receipt = {
                        "task": task,
                        "seed": seed,
                        "seed_formula": (
                            config["gumbel_seed_offset"]
                            + TASK_OFFSETS[task] + seed
                        ),
                        "schedule_seed": schedule_seed,
                        "shape": list(schedule.shape),
                        "dtype": str(schedule.dtype),
                        "fingerprint": tensor_fingerprint(schedule),
                        "path": schedule_name,
                        "sha256": sha256_file(schedule_path),
                    }
                    schedules[(task, seed)] = receipt
                    recorder.log("schedule", **receipt)

                    dataset = make_dataset(task, seed, config, shuffled=False)
                    dataset_name = f"main__{task}__s{seed}.pt"
                    atomic_torch_save(
                        run_dir / "datasets" / dataset_name, dataset
                    )
                    canonical_core, core_fingerprint = build_canonical_core(
                        task, seed, config
                    )
                    init_name = f"{task}__s{seed}.pt"
                    atomic_torch_save(
                        run_dir / "initializations" / init_name,
                        {
                            "task": task,
                            "seed": seed,
                            "core": canonical_core,
                            "fingerprint": core_fingerprint,
                        },
                    )
                    recorder.log(
                        "initialization",
                        task=task,
                        seed=seed,
                        fingerprint=core_fingerprint,
                        path=init_name,
                        sha256=sha256_file(
                            run_dir / "initializations" / init_name
                        ),
                    )

                    for arm in config["arms"]:
                        summary = run_cell(
                            task, arm, seed, False, steps, dataset, schedule,
                            receipt["fingerprint"], canonical_core,
                            core_fingerprint, config, device, run_dir,
                        )
                        summaries.append(summary)
                        recorder.log("cell_terminal", **summary)
                        write_json(
                            run_dir / "progress.json",
                            {
                                "completed_cells": [
                                    row["cell_id"] for row in summaries
                                ],
                                "expected_cells": 56,
                                "last_cell": summary,
                            },
                        )
                        lock_events.write({
                            "kind": "cell_heartbeat",
                            "cell_id": summary["cell_id"],
                            "time": time.time(),
                            "gpu": nvidia_sample(),
                        })

                    if task == config["shuffle_control"]["task"]:
                        shuffled = make_dataset(
                            task, seed, config, shuffled=True
                        )
                        control_name = f"control__{task}__s{seed}.pt"
                        atomic_torch_save(
                            run_dir / "datasets" / control_name, shuffled
                        )
                        for arm in config["shuffle_control"]["arms"]:
                            summary = run_cell(
                                task, arm, seed, True, steps, shuffled,
                                schedule, receipt["fingerprint"],
                                canonical_core, core_fingerprint, config,
                                device, run_dir,
                            )
                            summaries.append(summary)
                            recorder.log("cell_terminal", **summary)
                            write_json(
                                run_dir / "progress.json",
                                {
                                    "completed_cells": [
                                        row["cell_id"] for row in summaries
                                    ],
                                    "expected_cells": 56,
                                    "last_cell": summary,
                                },
                            )
                            lock_events.write({
                                "kind": "cell_heartbeat",
                                "cell_id": summary["cell_id"],
                                "time": time.time(),
                                "gpu": nvidia_sample(),
                            })

            integrity_failures = reduce_integrity(
                run_dir, summaries, schedules, source_receipts,
                registration_receipts, graph, clone_gate, config,
            )
            result_doc = score_predictions_and_verdict(
                summaries, graph, clone_gate, integrity_failures, config
            )
        else:
            result_doc = {
                "verdict": "INSTRUMENT_BROKEN",
                "predictions": {
                    "P1_graph_gate": {
                        "hit": False,
                        "graph": graph,
                        "deterministic_clone_gate": clone_gate,
                    }
                },
                "scores": {},
                "mechanisms": {},
                "integrity_failures": [],
            }
        verdict = result_doc["verdict"]
        summary_doc = {
            "graph_gate": graph,
            "deterministic_clone_gate": clone_gate,
            "cells": summaries,
            "schedules": list(schedules.values()),
            "result": result_doc,
        }
        verdict_doc = {
            **result_doc,
            "status": "PROVISIONAL_UNTIL_CHAIN_MANIFEST_AND_SEAL_VERIFY",
            "cell_count": len(summaries),
            "scope": (
                "two tiny random-point interpolation tasks, four unseen "
                "seeds, one architecture, one fixed heterogeneous assignment"
            ),
        }
        write_json(run_dir / "summary.json", summary_doc)
        write_json(run_dir / "verdict.json", verdict_doc)
        recorder.log("provisional_verdict", **verdict_doc)
        recorder.log(
            "execution_complete",
            verdict=verdict,
            cells=len(summaries),
            elapsed_seconds=round(time.time() - started, 3),
        )
    except BaseException as exc:
        verdict = "INCOMPLETE"
        error_doc = {
            "verdict": verdict,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "completed_cells": len(summaries),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(run_dir / "error.json", error_doc)
        write_json(
            run_dir / "summary.json",
            {"cells": summaries, "schedules": list(schedules.values())},
        )
        write_json(
            run_dir / "verdict.json",
            {
                "verdict": verdict,
                "status": "INTERRUPTED",
                "error": repr(exc),
                "cell_count": len(summaries),
            },
        )
        recorder.log("run_interrupted", **error_doc)
        raise
    finally:
        lock_events.write({
            "kind": "sealing",
            "time": time.time(),
            "verdict": verdict,
            "completed_cells": len(summaries),
        })
        lock_events.close()
        recorder.close()
        chain = verify_result_chain(run_dir / "results.jsonl")
        write_json(run_dir / "result_chain.json", chain)
        manifest = make_manifest(run_dir)
        write_json(run_dir / "manifest.json", manifest)
        manifest_hash = sha256_file(run_dir / "manifest.json")
        write_text(
            run_dir / "manifest.sha256",
            manifest_hash + "  manifest.json\n",
            encoding="ascii",
        )
        release_error = None
        if lock_acquired:
            try:
                gpu_lock.release()
            except BaseException as exc:
                release_error = repr(exc)
        write_json(
            run_dir / "gpu_lock_release.json",
            {
                "time": time.time(),
                "released": lock_acquired and release_error is None,
                "error": release_error,
                "gpu_uuid": EXPECTED_GPU_UUID,
            },
        )
        if release_error is not None:
            raise RuntimeError(release_error)

    completion = {
        "status": "PROVISIONAL_UNTIL_SEAL",
        "verdict": verdict,
        "manifest_sha256": manifest_hash,
        "result_chain": chain,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(run_dir / "completion.json", completion)
    seal = {
        "status": "FINAL",
        "verdict": verdict,
        "manifest_sha256": manifest_hash,
        "completion_sha256": sha256_file(run_dir / "completion.json"),
        "gpu_lock_release_sha256": sha256_file(
            run_dir / "gpu_lock_release.json"
        ),
        "result_chain": chain,
    }
    write_json(run_dir / "seal.json", seal)
    print(f"E58 complete: {verdict} in {(time.time() - started) / 60:.1f} min")
    print(f"manifest_sha256={manifest_hash}")
    print(f"seal_sha256={sha256_file(run_dir / 'seal.json')}")
    return 0 if verdict not in {
        "INCOMPLETE", "INSTRUMENT_BROKEN", "INVALID_LEAKAGE_CONTROL"
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
