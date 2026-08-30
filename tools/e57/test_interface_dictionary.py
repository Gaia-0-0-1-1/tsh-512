"""Synthetic/instrument tests for E57; no registered component training."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import interface_dictionary as e57  # noqa: E402


class AlgebraTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(e57.CONFIG_PATH.read_text(encoding="utf-8"))

    def test_registered_permutation_is_deranged_bijection(self):
        pi, inv = e57.permutation_tensors(self.config)
        self.assertEqual(sorted(pi.tolist()), list(range(8)))
        self.assertTrue(torch.all(pi != torch.arange(8)))
        self.assertTrue(torch.equal(inv[pi], torch.arange(8)))

    def test_explicit_and_gauge_repairs_restore_mixture(self):
        torch.manual_seed(91)
        p = torch.softmax(torch.randn(17, 8), dim=-1)
        embeddings = torch.randn(8, 13)
        pi, inv = e57.permutation_tensors(self.config)
        q = p[:, inv]
        baseline = p @ embeddings
        explicit = q[:, pi] @ embeddings
        gauge = q @ embeddings[inv]
        self.assertTrue(torch.equal(q[:, pi], p))
        self.assertTrue(torch.equal(explicit, baseline))
        self.assertTrue(torch.allclose(
            gauge, baseline,
            rtol=self.config["rtol"], atol=self.config["atol"],
        ))


class InstrumentTests(unittest.TestCase):
    def test_component_and_composite_domains_are_complete(self):
        for task in ("Z4xZ2", "Z2x2x2"):
            x, y = e57.full_component_domain(task)
            self.assertEqual(tuple(x.shape), (64, 2))
            self.assertEqual(tuple(y.shape), (64,))
        for outer, inner in (("Z4xZ2", "Z2x2x2"),
                             ("Z2x2x2", "Z4xZ2")):
            ds = e57.composite_domain(outer, inner)
            self.assertEqual(tuple(ds["x"].shape), (4096, 4))
            self.assertEqual(torch.bincount(ds["y"], minlength=8).tolist(),
                             [512] * 8)

    def test_b_stack_is_faithful_on_random_model(self):
        torch.manual_seed(7)
        model = e57.TinyTransformer(8, 8, d=64, lattice="phi1").eval()
        x = torch.tensor([[a, b] for a in range(8) for b in range(8)])
        with torch.no_grad():
            direct = model(x)
            entered = e57.b_stack(model, model.embed(x))
        self.assertLess((direct - entered).abs().max().item(), 1e-5)

    def test_config_digest_is_embedded(self):
        self.assertEqual(e57.sha256_file(e57.CONFIG_PATH),
                         e57.EXPECTED_CONFIG_SHA256)

    def test_result_chain_detects_valid_records(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "results.jsonl"
            recorder = e57.ChainedRecorder(path)
            recorder.log("a", value=1)
            recorder.log("b", value=2)
            recorder.close()
            receipt = e57.verify_result_chain(path)
        self.assertEqual(receipt["records"], 2)
        self.assertNotEqual(receipt["head"], "0" * 64)

    def test_canonical_timeline_chain_and_one_based_sequence(self):
        records = e57.timeline_records()
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["seq"], 1)
        self.assertEqual(records[-1]["seq"], len(records))

    def test_result_chain_detects_tamper(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "results.jsonl"
            recorder = e57.ChainedRecorder(path)
            recorder.log("a", value=1)
            recorder.close()
            record = json.loads(path.read_text(encoding="utf-8"))
            record["value"] = 2
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                e57.verify_result_chain(path)

    def test_registration_hash_verifier_fails_on_tamper(self):
        relative = "tools/e57/preregistered_config.json"
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "registration.json"
            path.write_text(
                json.dumps({"sha256": {relative: "0" * 64}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                e57.verify_registered_hashes(path, {relative})

    def test_dedicated_batch_generator_replays_legacy_rng_stream(self):
        seed = 19
        torch.manual_seed(seed)
        e57.TinyTransformer(8, 8, d=64, lattice="phi1")
        generator = torch.Generator(device="cpu")
        generator.set_state(torch.get_rng_state().clone())
        expected = torch.randperm(51)
        observed = torch.randperm(51, generator=generator)
        self.assertTrue(torch.equal(observed, expected))

    def test_state_fingerprint_detects_mutation(self):
        torch.manual_seed(23)
        model = e57.TinyTransformer(8, 8, d=64, lattice="phi1")
        before = e57.state_dict_fingerprint(model)
        with torch.no_grad():
            next(model.parameters()).view(-1)[0].add_(1)
        self.assertNotEqual(before, e57.state_dict_fingerprint(model))

    def test_verdict_precedence_does_not_mask_core_failure(self):
        buckets = {
            "integrity": [],
            "core": ["baseline mismatch"],
            "gauge": ["gauge drift"],
            "break": ["break missing"],
        }
        self.assertEqual(e57.select_verdict(True, buckets), "INSTRUMENT_FAIL")
        buckets["integrity"].append("cell missing")
        self.assertEqual(e57.select_verdict(True, buckets), "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
