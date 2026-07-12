import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from data.generate import (
    accuracy_v2_gate_result,
    dedupe_work_items,
    eval_prompt_keys,
    record_passes_accuracy_gate,
)
from data.audit_v4r3 import audit
from data.v4r3 import norm_text, target_config
from data.export_clean_replay import clean_replay_records
from eval.tuned_sweep import validate_eval_policy


REPO_ROOT = Path(__file__).resolve().parent.parent


class ReservedPromptTests(unittest.TestCase):
    def setUp(self):
        self.concepts = json.loads(
            (REPO_ROOT / "data" / "concepts.json").read_text(encoding="utf-8")
        )

    def test_eval_calibration_and_blind_prompts_are_disjoint_and_reserved(self):
        eval_prompts = {norm_text(item) for item in self.concepts["eval"]}
        calibration = {
            norm_text(item) for item in self.concepts["calibration_v4r5"]
        }
        blind = {norm_text(item) for item in self.concepts["blind_v4r5"]}
        self.assertEqual(len(eval_prompts), 24)
        self.assertEqual(len(calibration), 24)
        self.assertEqual(len(blind), 24)
        self.assertTrue(eval_prompts.isdisjoint(calibration))
        self.assertTrue(eval_prompts.isdisjoint(blind))
        self.assertTrue(calibration.isdisjoint(blind))
        self.assertEqual(
            eval_prompt_keys(self.concepts), eval_prompts | calibration | blind
        )

    def test_calibration_prompt_is_removed_from_training_work(self):
        prompt = self.concepts["calibration_v4r5"][0]
        items = [{"concept": prompt, "phrasing": prompt}]
        kept = dedupe_work_items(items, set(), eval_prompt_keys(self.concepts))
        self.assertEqual(kept, [])

    def test_new_prompt_sets_do_not_exactly_match_v4r4_training_prompts(self):
        training_keys = set()
        dataset_path = REPO_ROOT / "data" / "v4" / "gold_v4_r4.jsonl"
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            training_keys.add(norm_text(record.get("phrasing") or record["concept"]))
        reserved = {
            norm_text(prompt)
            for key in ("calibration_v4r5", "blind_v4r5")
            for prompt in self.concepts[key]
        }
        self.assertTrue(training_keys.isdisjoint(reserved))


class FrozenEvalPolicyTests(unittest.TestCase):
    def test_heldout_requires_explicit_final_eval(self):
        with self.assertRaisesRegex(ValueError, "require --final-eval"):
            validate_eval_policy("eval_litmus", [0.3], [0], 1, False)

    def test_heldout_rejects_temperature_selection(self):
        with self.assertRaisesRegex(ValueError, "one preselected temperature"):
            validate_eval_policy("eval_litmus", [0.0, 0.3], [0], 2, True)

    def test_heldout_rejects_seed_filtering(self):
        with self.assertRaisesRegex(ValueError, "keep every setting"):
            validate_eval_policy("eval_litmus", [0.3], [0, 1, 2], 2, True)

    def test_heldout_accepts_one_fixed_complete_run(self):
        validate_eval_policy("eval_litmus", [0.3], [0, 1, 2], 3, True)

    def test_calibration_cannot_be_marked_final(self):
        with self.assertRaisesRegex(ValueError, "only valid for held-out"):
            validate_eval_policy("calibration_v4r5", [0.0, 0.3], [0, 1], 2, True)

    def test_blind_holdout_uses_frozen_eval_policy(self):
        validate_eval_policy("blind_v4r5", [0.0], [0], 1, True)


class GenerationAccuracyGateTests(unittest.TestCase):
    def setUp(self):
        self.primary = ["gpt", "claude"]
        self.tiebreaker = "gemini"
        self.clean = {
            "factuality": 3,
            "mechanism": 2,
            "errors": [],
            "justification": "clean",
        }
        self.minor = {
            "factuality": 2,
            "mechanism": 2,
            "errors": [
                {
                    "severity": "minor",
                    "claim": "small imprecision",
                    "correction": "Use the precise local wording.",
                }
            ],
            "justification": "minor only",
        }

    def test_clean_gate_rejects_consensus_minor_error(self):
        result = accuracy_v2_gate_result(
            {"gpt": self.minor, "claude": self.minor},
            self.primary,
            self.tiebreaker,
            require_clean=True,
        )
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["consensus"]["clean_pass"])
        self.assertIn("precise local wording", result["note"])

    def test_tolerant_gate_accepts_consensus_minor_error(self):
        result = accuracy_v2_gate_result(
            {"gpt": self.minor, "claude": self.minor},
            self.primary,
            self.tiebreaker,
            require_clean=False,
        )
        self.assertEqual(result["score"], 2)
        self.assertTrue(result["consensus"]["accuracy_pass_v2"])

    def test_clean_v2_seed_requires_saved_clean_consensus(self):
        legacy = {"accuracy": {"score": 2}}
        clean = {"accuracy": {"consensus": {"clean_pass": True}}}
        self.assertFalse(record_passes_accuracy_gate(legacy, "clean-v2"))
        self.assertTrue(record_passes_accuracy_gate(clean, "clean-v2"))

    def test_replay_export_keeps_only_clean_consensus(self):
        base = {
            "concept": "prompt",
            "text": "answer",
            "source": "source.jsonl",
            "readability_pass": True,
            "training_metadata": {"concept": "canonical"},
            "judgments": {"gpt": self.clean, "claude": self.clean},
        }
        data = {
            "rubric_version": "accuracy_v2",
            "records": [
                {**base, "consensus": {"clean_pass": True}},
                {**base, "concept": "minor", "consensus": {"clean_pass": False}},
            ],
        }
        replay = clean_replay_records(data)
        self.assertEqual(len(replay), 1)
        self.assertEqual(replay[0]["phrasing"], "prompt")
        self.assertTrue(replay[0]["accuracy"]["consensus"]["clean_pass"])


class V4R6MixtureTests(unittest.TestCase):
    def test_frozen_mixture_has_exact_clean_composition(self):
        dataset_path = REPO_ROOT / "data" / "v4" / "gold_v4_r6.jsonl"
        stats_path = REPO_ROOT / "data" / "v4" / "gold_v4_r6.stats.json"
        records = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        content = dataset_path.read_bytes().replace(b"\r\n", b"\n").replace(
            b"\r", b"\n"
        )
        digest = hashlib.sha256(content).hexdigest()

        self.assertEqual(digest, stats["dataset_sha256"])
        self.assertEqual(len(records), 400)
        self.assertEqual(
            Counter(record["mixture_source"] for record in records),
            {
                "v4r2_accuracy_anchor": 98,
                "v4r4_readability_replay": 102,
                "v4r5_clean_target": 200,
            },
        )
        summary = audit(
            dataset_path,
            target_config(),
            min_sentences=4,
            max_sentences=6,
            accuracy_gate="clean-v2",
            forbid_targeted_v4r3=True,
        )
        self.assertTrue(summary["passed"], summary["failures"])

    def test_v4r7_clean_union_has_exact_composition(self):
        dataset_path = REPO_ROOT / "data" / "v4" / "gold_v4_r7.jsonl"
        stats_path = REPO_ROOT / "data" / "v4" / "gold_v4_r7.stats.json"
        records = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        content = dataset_path.read_bytes().replace(b"\r\n", b"\n").replace(
            b"\r", b"\n"
        )

        self.assertEqual(hashlib.sha256(content).hexdigest(), stats["dataset_sha256"])
        self.assertEqual(len(records), 485)
        self.assertEqual(
            Counter(record["mixture_source"] for record in records),
            {
                "v4r2_accuracy_anchor": 98,
                "v4r4_readability_replay": 106,
                "v4r5_clean_target": 281,
            },
        )
        summary = audit(
            dataset_path,
            target_config(),
            min_sentences=4,
            max_sentences=6,
            accuracy_gate="clean-v2",
            forbid_targeted_v4r3=True,
        )
        self.assertTrue(summary["passed"], summary["failures"])


if __name__ == "__main__":
    unittest.main()
