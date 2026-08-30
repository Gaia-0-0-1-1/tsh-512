"""Fast, CPU-only unit tests for E56 data and gate mechanics."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wall_falsifier as wf  # noqa: E402


class DomainTests(unittest.TestCase):
    def test_all_registered_domains_are_total_and_balanced(self):
        for task in (
            *wf.HARD_TASKS,
            "T8(T8)",
            "SHUFFLED_Z4xZ2(Z4xZ2)",
        ):
            x, y = wf.full_domain(task)
            self.assertEqual(tuple(x.shape), (4096, 4))
            self.assertEqual(tuple(y.shape), (4096,))
            self.assertTrue(torch.all((0 <= y) & (y < 8)))
            self.assertEqual(torch.bincount(y, minlength=8).tolist(),
                             [512] * 8)

    def test_shuffle_is_fixed_and_preserves_histogram(self):
        _, base = wf.full_domain("Z4xZ2(Z4xZ2)")
        _, shuffled_a = wf.full_domain("SHUFFLED_Z4xZ2(Z4xZ2)")
        _, shuffled_b = wf.full_domain("SHUFFLED_Z4xZ2(Z4xZ2)")
        self.assertTrue(torch.equal(shuffled_a, shuffled_b))
        self.assertFalse(torch.equal(base, shuffled_a))
        self.assertEqual(torch.bincount(base, minlength=8).tolist(),
                         torch.bincount(shuffled_a, minlength=8).tolist())

    def test_split_fingerprint_is_reproducible(self):
        a = wf.make_data(wf.HARD_TASKS[0], 0, 0.8, torch.device("cpu"))
        b = wf.make_data(wf.HARD_TASKS[0], 0, 0.8, torch.device("cpu"))
        c = wf.make_data(wf.HARD_TASKS[0], 1, 0.8, torch.device("cpu"))
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        self.assertNotEqual(a["fingerprint"], c["fingerprint"])
        self.assertEqual((a["train_size"], a["test_size"]), (3276, 820))


class GateTests(unittest.TestCase):
    def test_gate_requires_consecutive_evaluations(self):
        gate = wf.SustainedGate(required=3)
        self.assertFalse(gate.update(True, 200))
        self.assertFalse(gate.update(False, 400))
        self.assertFalse(gate.update(True, 600))
        self.assertFalse(gate.update(True, 800))
        self.assertTrue(gate.update(True, 1000))
        self.assertEqual(gate.first_step, 600)
        self.assertEqual(gate.confirmed_step, 1000)

    def test_classification_distinguishes_delayed_grok(self):
        config = {"eval_every": 200, "sustain_evaluations": 5}
        memorize = wf.SustainedGate(5, first_step=1000, confirmed_step=1800)
        grok = wf.SustainedGate(5, first_step=5000, confirmed_step=5800)
        self.assertEqual(wf.classify(memorize, grok, config),
                         "POST_MEMORIZATION_GROK")
        close = copy.deepcopy(grok)
        close.first_step = 1800
        close.confirmed_step = 2600
        self.assertEqual(wf.classify(memorize, close, config),
                         "ORDINARY_GENERALIZATION")

    def test_window_boundary_is_four_intervals(self):
        config = {"eval_every": 200, "sustain_evaluations": 5}
        memorize = wf.SustainedGate(5, first_step=1000, confirmed_step=1800)
        edge = wf.SustainedGate(5, first_step=1800, confirmed_step=2600)
        late = wf.SustainedGate(5, first_step=2000, confirmed_step=2800)
        self.assertEqual(wf.classify(memorize, edge, config),
                         "ORDINARY_GENERALIZATION")
        self.assertEqual(wf.classify(memorize, late, config),
                         "POST_MEMORIZATION_GROK")

    def test_terminal_forgetting_is_explicit(self):
        config = {
            "eval_every": 200,
            "sustain_evaluations": 5,
            "memorize_threshold": 0.995,
            "grok_threshold": 0.95,
        }
        memorize = wf.SustainedGate(5, first_step=1000, confirmed_step=1800)
        grok = wf.SustainedGate(5, first_step=5000, confirmed_step=5800)
        status = wf.classify_terminal(
            memorize, grok, config, {"train_acc": 0.98, "test_acc": 0.97}
        )
        self.assertEqual(status, "POST_MEMORIZATION_GROK_THEN_FORGOT")


class BranchAndVerdictTests(unittest.TestCase):
    def test_clone_preserves_model_optimizer_and_batch_stream(self):
        config = json.loads(wf.CONFIG_PATH.read_text(encoding="utf-8"))
        device = torch.device("cpu")
        wf.seed_model(3)
        model = wf.make_model(config["width"], device)
        optimizer = wf.make_optimizer(model, config, 0.0)
        generator = wf.make_batch_generator(device, 3)
        x = torch.randint(0, 8, (8, 4))
        y = torch.randint(0, 8, (8,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

        clone, clone_opt, clone_gen = wf.clone_branch(
            model, optimizer, generator, config, device, 0.5
        )
        for original, copied in zip(model.parameters(), clone.parameters()):
            self.assertTrue(torch.equal(original, copied))
        self.assertEqual(clone_opt.param_groups[0]["weight_decay"], 0.5)
        self.assertTrue(torch.equal(generator.get_state(), clone_gen.get_state()))
        self.assertTrue(torch.equal(
            torch.randperm(100, generator=generator),
            torch.randperm(100, generator=clone_gen),
        ))

    def test_no_memorization_path_emits_terminal_summary(self):
        config = json.loads(wf.CONFIG_PATH.read_text(encoding="utf-8"))
        config["phase_a_steps"] = 2
        config["eval_every"] = 1
        config["sustain_evaluations"] = 2
        config["memorize_threshold"] = 1.1
        device = torch.device("cpu")
        data = wf.make_data(wf.HARD_TASKS[0], 0, 0.8, device)
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            (run_dir / "checkpoints").mkdir()
            recorder = wf.JsonlRecorder(run_dir / "results.jsonl")
            rows = wf.run_fork_assay(
                wf.HARD_TASKS[0], 0, data, config, device, run_dir, recorder
            )
            recorder.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["classification"],
                         "NO_MEMORIZATION_WITHIN_BUDGET")

    def test_shuffled_generalization_invalidates_aggregate(self):
        config = json.loads(wf.CONFIG_PATH.read_text(encoding="utf-8"))
        rows = []
        for task in (*wf.HARD_TASKS, "T8(T8)",
                     "SHUFFLED_Z4xZ2(Z4xZ2)"):
            for seed in config["seeds"]:
                generalized = task == "SHUFFLED_Z4xZ2(Z4xZ2)" and seed == 0
                rows.append({
                    "run_key": f"fork::{task}::seed={seed}",
                    "task": task,
                    "seed": seed,
                    "phase": "phase_a",
                    "memorization_confirmed_step": 1000,
                    "grok_confirmed_step": 2000 if generalized else None,
                })
        for task in config["sentinel_tasks"]:
            for seed in config["seeds"]:
                rows.append({
                    "run_key": f"sentinel::{task}::seed={seed}",
                    "task": task,
                    "seed": seed,
                    "phase": "sentinel",
                    "memorization_confirmed_step": None,
                    "grok_confirmed_step": None,
                    "classification": "NO_MEMORIZATION_WITHIN_BUDGET",
                })
        verdict = wf.aggregate_verdict(rows, config)
        self.assertEqual(verdict["score"], "INVALID_SHUFFLED_CONTROL")
        self.assertTrue(verdict["shuffled_control"]["invalidates_assay"])

    def test_missing_hard_cells_fail_closed(self):
        config = json.loads(wf.CONFIG_PATH.read_text(encoding="utf-8"))
        verdict = wf.aggregate_verdict([], config)
        self.assertEqual(verdict["score"], "INCOMPLETE")
        self.assertEqual(len(verdict["missing_hard_cells"]), 8)
        self.assertEqual(len(verdict["missing_fork_cells"]), 12)
        self.assertEqual(len(verdict["missing_sentinel_cells"]), 8)

    def test_missing_control_cells_fail_closed(self):
        config = json.loads(wf.CONFIG_PATH.read_text(encoding="utf-8"))
        rows = []
        for task in wf.HARD_TASKS:
            for seed in config["seeds"]:
                rows.append({
                    "run_key": f"fork::{task}::seed={seed}",
                    "task": task,
                    "seed": seed,
                    "phase": "phase_a",
                    "memorization_confirmed_step": None,
                    "grok_confirmed_step": None,
                })
                rows.append({
                    "run_key": f"sentinel::{task}::seed={seed}",
                    "task": task,
                    "seed": seed,
                    "phase": "sentinel",
                    "memorization_confirmed_step": None,
                    "grok_confirmed_step": None,
                    "classification": "NO_MEMORIZATION_WITHIN_BUDGET",
                })
        verdict = wf.aggregate_verdict(rows, config)
        self.assertEqual(verdict["score"], "INCOMPLETE")
        self.assertEqual(len(verdict["missing_fork_cells"]), 4)


if __name__ == "__main__":
    unittest.main()
