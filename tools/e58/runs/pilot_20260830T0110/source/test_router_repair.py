"""Implementation tests for E58. No registered performance training."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.e58 import router_repair as e58


def empty_gate(mode, mem=1000, gen=None):
    return {
        mode: {
            "memorization": {
                "onset_step": mem,
                "confirmation_step": None if mem is None else mem + 800,
            },
            "generalization": {
                "onset_step": gen,
                "confirmation_step": None if gen is None else gen + 800,
            },
            "delay_steps": None if mem is None or gen is None else gen - mem,
            "classification": "NO_SUSTAINED_GENERALIZATION",
            "terminal": {},
        }
    }


def synthetic_summaries(config):
    rows = []
    for task in config["tasks"]:
        for seed in config["seeds"]:
            for arm in config["arms"]:
                modes = {}
                for mode in e58.evaluation_modes(arm):
                    modes.update(empty_gate(mode))
                rows.append({
                    "cell_id": e58.cell_id(task, arm, seed, False),
                    "task": task,
                    "arm": arm,
                    "seed": seed,
                    "control": False,
                    "gates": modes,
                    "terminal_route_diversity": 1,
                    "primary_generalization_route_diversity": None,
                })
    task = config["shuffle_control"]["task"]
    for seed in config["seeds"]:
        for arm in config["shuffle_control"]["arms"]:
            modes = {}
            for mode in e58.evaluation_modes(arm):
                modes.update(empty_gate(mode))
            rows.append({
                "cell_id": e58.cell_id(task, arm, seed, True),
                "task": task,
                "arm": arm,
                "seed": seed,
                "control": True,
                "gates": modes,
                "terminal_route_diversity": 1,
                "primary_generalization_route_diversity": None,
            })
    return rows


def set_z8_generalization(rows, arm, mode, step, diversity=1):
    for row in rows:
        if not row["control"] and row["task"] == "Z8" and row["arm"] == arm:
            row["gates"][mode]["generalization"] = {
                "onset_step": step,
                "confirmation_step": step + 800,
            }
            row["terminal_route_diversity"] = diversity
            row["primary_generalization_route_diversity"] = diversity


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(e58.CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_digest_and_grid_size(self):
        self.assertEqual(e58.sha256_file(e58.CONFIG_PATH), e58.EXPECTED_CONFIG_SHA256)
        self.assertEqual(len(e58.expected_cell_specs(self.config)), 56)

    def test_all_arms_have_equal_nominal_parameter_count(self):
        for arm in self.config["arms"]:
            model = e58.RouterTransformer(
                8, 8, self.config["width"], arm,
                self.config["heterogeneous_assignment"],
                self.config["shadow_parameters"],
            )
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                self.config["nominal_parameters"],
            )

    def test_core_initialization_is_identical_across_arms(self):
        state, fingerprint = e58.build_canonical_core("Z8", 5700, self.config)
        for arm in self.config["arms"]:
            model = e58.build_model(
                "Z8", 5700, arm, self.config, state,
                torch.device("cpu"),
            )
            self.assertEqual(e58.mapping_fingerprint(e58.core_state(model)), fingerprint)

    def test_registered_gumbel_schedule_is_deterministic(self):
        first, seed_a = e58.gumbel_schedule("Z8", 5700, 11, self.config)
        second, seed_b = e58.gumbel_schedule("Z8", 5700, 11, self.config)
        other, _ = e58.gumbel_schedule("Z2x2x2", 5700, 11, self.config)
        self.assertEqual(seed_a, seed_b)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(tuple(first.shape), (11, 7, 3))

    def test_shuffle_is_nonidentity_and_histogram_preserving(self):
        raw = e58.make_dataset("Z2x2x2", 5700, self.config, shuffled=False)
        shuffled = e58.make_dataset("Z2x2x2", 5700, self.config, shuffled=True)
        permutation = shuffled["shuffle_permutation"]
        self.assertFalse(torch.equal(permutation, torch.arange(64)))
        self.assertTrue(torch.equal(
            torch.bincount(raw["y"], minlength=8),
            torch.bincount(shuffled["y"], minlength=8),
        ))
        self.assertTrue(torch.equal(raw["train_indices"], shuffled["train_indices"]))

    def test_graph_repair_gate_on_cpu(self):
        gate = e58.graph_gate(self.config, torch.device("cpu"))
        self.assertTrue(gate["passed"], gate)
        self.assertTrue(gate["forward_bit_identical"])
        self.assertTrue(gate["core_gradients_bit_identical"])
        self.assertTrue(gate["post_step_core_bit_identical"])

    def test_deterministic_clone_gate_on_cpu(self):
        gate = e58.deterministic_clone_gate(self.config, torch.device("cpu"))
        self.assertTrue(gate["passed"], gate)

    def test_zero_logit_hard_tie_selects_ternary(self):
        torch.manual_seed(2)
        layer = e58.RoutedLinear(5, 4, "dead")
        hard = layer._mixture("hard", None, 1.0)
        self.assertTrue(torch.equal(hard, layer._quantized("ternary")))

    def test_sustained_gate_and_grokking_classification(self):
        curves = []
        for step in range(0, 2401, 200):
            curves.append({
                "step": step,
                "modes": {
                    "native": {
                        "train_accuracy": 1.0 if step >= 200 else 0.0,
                        "test_accuracy": 1.0 if step >= 1400 else 0.0,
                    }
                },
            })
        summary = e58.summarize_curves(curves, "FP", self.config)["native"]
        self.assertEqual(summary["memorization"]["onset_step"], 200)
        self.assertEqual(summary["generalization"]["onset_step"], 1400)
        self.assertEqual(summary["classification"], "DELAYED_GROKKING")

    def test_test_gate_before_memorization_is_never_grokking(self):
        curves = []
        for step in range(0, 2401, 200):
            curves.append({
                "step": step,
                "modes": {
                    "native": {
                        "train_accuracy": 1.0 if step >= 1400 else 0.0,
                        "test_accuracy": 1.0 if step >= 200 else 0.0,
                    }
                },
            })
        summary = e58.summarize_curves(curves, "FP", self.config)["native"]
        self.assertEqual(
            summary["classification"],
            "GENERALIZED_WITHOUT_MEMORIZATION_GATE",
        )

    def test_short_cpu_cell_closes_artifacts(self):
        config = copy.deepcopy(self.config)
        config["eval_every"] = 1
        task, seed, steps = "Z2x2x2", 5700, 2
        dataset = e58.make_dataset(task, seed, config)
        schedule, _ = e58.gumbel_schedule(task, seed, steps, config)
        state, fingerprint = e58.build_canonical_core(task, seed, config)
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            for name in ("cells", "curves", "checkpoints"):
                (run_dir / name).mkdir()
            summary = e58.run_cell(
                task, "FP", seed, False, steps, dataset, schedule,
                e58.tensor_fingerprint(schedule), state, fingerprint,
                config, torch.device("cpu"), run_dir,
            )
            self.assertEqual(summary["evaluations"], 3)
            self.assertTrue((run_dir / "cells" / summary["summary"]).is_file())
            self.assertTrue(
                (run_dir / "checkpoints" / summary["checkpoint"]).is_file()
            )

    def test_result_chain_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "results.jsonl"
            recorder = e58.ChainedRecorder(path)
            recorder.log("x", value=1)
            recorder.close()
            row = json.loads(path.read_text(encoding="utf-8"))
            row["value"] = 2
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                e58.verify_result_chain(path)

    def test_windows_gpu_lock_is_exclusive_and_recoverable(self):
        with tempfile.TemporaryDirectory() as raw, mock.patch.dict(
            os.environ, {"CODEX_HOME": raw}
        ):
            first = e58.WindowsExclusiveGpuLock("GPU-test")
            second = e58.WindowsExclusiveGpuLock("GPU-test")
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "lock unavailable"):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()


class VerdictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(e58.CONFIG_PATH.read_text(encoding="utf-8"))
        cls.good_graph = {"passed": True}

    def reduce(self, rows, graph=None, integrity=None):
        return e58.score_predictions_and_verdict(
            rows,
            self.good_graph if graph is None else graph,
            self.good_graph,
            [] if integrity is None else integrity,
            self.config,
        )

    def adaptive_rows(self, diversity=2):
        rows = synthetic_summaries(self.config)
        for arm, mode, step in (
            ("LIVE_WD0", "hard", 4000),
            ("LIVE_WD0", "soft", 4000),
            ("LEGACY_DEAD", "hard", 7000),
            ("LEGACY_DEAD", "soft", 7000),
            ("PHI1", "native", 7000),
            ("HETERO_FIXED", "native", 7000),
        ):
            set_z8_generalization(rows, arm, mode, step, diversity)
        return rows

    def test_all_nine_verdict_branches(self):
        adaptive = self.adaptive_rows(diversity=2)
        self.assertEqual(self.reduce(adaptive)["verdict"], "ADAPTIVE_ROUTER_BENEFIT")
        global_rows = self.adaptive_rows(diversity=1)
        self.assertEqual(self.reduce(global_rows)["verdict"], "GLOBAL_CHOICE_BENEFIT")

        mixture = synthetic_summaries(self.config)
        for arm, mode, step in (
            ("LIVE_WD0", "soft", 4000),
            ("LEGACY_DEAD", "soft", 7000),
            ("PHI1", "native", 7000),
            ("HETERO_FIXED", "native", 7000),
        ):
            set_z8_generalization(mixture, arm, mode, step)
        self.assertEqual(self.reduce(mixture)["verdict"], "MIXTURE_ONLY_BENEFIT")

        hetero = synthetic_summaries(self.config)
        set_z8_generalization(hetero, "HETERO_FIXED", "native", 4000)
        set_z8_generalization(hetero, "PHI1", "native", 7000)
        self.assertEqual(
            self.reduce(hetero)["verdict"],
            "HETEROGENEOUS_REPRESENTATION_ONLY",
        )

        neutral = synthetic_summaries(self.config)
        self.assertEqual(self.reduce(neutral)["verdict"], "NO_BOUNDED_ROUTER_BENEFIT")
        self.assertEqual(
            self.reduce(neutral, graph={"passed": False})["verdict"],
            "INSTRUMENT_BROKEN",
        )
        self.assertEqual(
            self.reduce(neutral, integrity=["missing"])["verdict"],
            "INCOMPLETE",
        )

        invalid = synthetic_summaries(self.config)
        control = next(row for row in invalid if row["control"])
        mode = e58.primary_mode(control["arm"])
        control["gates"][mode]["generalization"]["confirmation_step"] = 2000
        self.assertEqual(
            self.reduce(invalid)["verdict"], "INVALID_LEAKAGE_CONTROL"
        )

        soft_invalid = synthetic_summaries(self.config)
        live_control = next(
            row for row in soft_invalid
            if row["control"] and row["arm"] == "LIVE_WD0"
        )
        live_control["gates"]["soft"]["generalization"][
            "confirmation_step"
        ] = 2200
        self.assertEqual(
            self.reduce(soft_invalid)["verdict"],
            "INVALID_LEAKAGE_CONTROL",
        )

        floor = synthetic_summaries(self.config)
        for row in floor:
            if not row["control"] and row["task"] == "Z2x2x2":
                mode = e58.primary_mode(row["arm"])
                row["gates"][mode]["memorization"]["confirmation_step"] = None
        self.assertEqual(self.reduce(floor)["verdict"], "OPTIMIZATION_FLOOR")

    def test_p2_can_hit_without_top_fixed_comparator_benefit(self):
        rows = synthetic_summaries(self.config)
        set_z8_generalization(rows, "LIVE_WD0", "hard", 5000, diversity=2)
        set_z8_generalization(rows, "LEGACY_DEAD", "hard", 8000)
        set_z8_generalization(rows, "PHI1", "native", 3000)
        set_z8_generalization(rows, "HETERO_FIXED", "native", 3000)
        result = self.reduce(rows)
        self.assertTrue(result["predictions"]["P2_live_hard_beats_dead_hard"]["hit"])
        self.assertEqual(result["verdict"], "NO_BOUNDED_ROUTER_BENEFIT")

    def test_mechanism_verdicts_respect_full_comparators_and_floor_bars(self):
        mixture = synthetic_summaries(self.config)
        for arm, mode, step in (
            ("LIVE_WD0", "soft", 3000),
            ("LIVE_WD0", "hard", 4000),
            ("LEGACY_DEAD", "soft", 7000),
            ("LEGACY_DEAD", "hard", 3000),
            ("PHI1", "native", 7000),
            ("HETERO_FIXED", "native", 7000),
        ):
            set_z8_generalization(mixture, arm, mode, step, diversity=2)
        # Hard beats both fixed arms but loses the dead comparator; only the
        # soft path clears all three comparators.
        self.assertEqual(self.reduce(mixture)["verdict"], "MIXTURE_ONLY_BENEFIT")

        dead_floor_fail = copy.deepcopy(mixture)
        for row in dead_floor_fail:
            if (
                not row["control"] and row["task"] == "Z2x2x2"
                and row["arm"] == "LEGACY_DEAD"
            ):
                row["gates"]["hard"]["memorization"]["confirmation_step"] = None
        self.assertEqual(
            self.reduce(dead_floor_fail)["verdict"],
            "NO_BOUNDED_ROUTER_BENEFIT",
        )

        tied = synthetic_summaries(self.config)
        for arm, mode in (
            ("LIVE_WD0", "hard"),
            ("LIVE_WD0", "soft"),
            ("HETERO_FIXED", "native"),
        ):
            set_z8_generalization(tied, arm, mode, 4000)
        set_z8_generalization(tied, "PHI1", "native", 7000)
        self.assertEqual(self.reduce(tied)["verdict"], "NO_BOUNDED_ROUTER_BENEFIT")


if __name__ == "__main__":
    unittest.main()
