import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from litmus.accuracy_v2 import (
    accuracy_pass_v2,
    build_consensus,
    clean_pass,
    validate_judgment,
)
from litmus.benchmark_v2 import MODEL_ORDER, load_benchmark_records
from litmus.concepts import CONCEPTS
from litmus.judge_accuracy_v2 import (
    finalize_consensus,
    judge_one,
    load_input_records,
    prepare_output,
    preflight_models,
    tiebreaker_tasks,
)
from eval.summarize_sweep_v2 import summarize
from eval.tuned_sweep import readability_penalty, setting_key
from litmus.report_accuracy_v2 import (
    HISTORICAL_START,
    V2_START,
    aggregate,
    install_versioned_report,
)


def judgment(factuality=3, mechanism=2):
    errors = []
    if factuality == 2:
        errors = [
            {
                "severity": "minor",
                "claim": "small imprecision",
                "correction": "more precise wording",
            }
        ]
    elif factuality <= 1:
        errors = [
            {
                "severity": "major",
                "claim": "wrong causal claim",
                "correction": "correct causal claim",
            }
        ]
    return validate_judgment(
        {
            "factuality": factuality,
            "mechanism": mechanism,
            "errors": errors,
            "justification": "fixture judgment",
        }
    )


class AccuracyV2RubricTests(unittest.TestCase):
    def test_clean_and_minor_error_passes_are_distinct(self):
        self.assertTrue(clean_pass(3, 2))
        self.assertTrue(accuracy_pass_v2(3, 2))
        self.assertFalse(clean_pass(2, 2))
        self.assertTrue(accuracy_pass_v2(2, 2))
        self.assertFalse(accuracy_pass_v2(3, 1))

    def test_validation_enforces_score_error_consistency(self):
        self.assertEqual(judgment(3, 2)["errors"], [])
        self.assertEqual(judgment(2, 2)["errors"][0]["severity"], "minor")
        self.assertEqual(judgment(1, 2)["errors"][0]["severity"], "major")
        with self.assertRaises(ValueError):
            validate_judgment(
                {
                    "factuality": 3,
                    "mechanism": 2,
                    "errors": [{"severity": "minor", "claim": "x", "correction": "y"}],
                    "justification": "invalid",
                }
            )

    def test_exact_agreement_never_uses_tiebreaker(self):
        result = build_consensus(judgment(3, 2), judgment(3, 2))
        self.assertEqual(result["factuality"], 3)
        self.assertFalse(result["tiebreaker_used"])
        with self.assertRaises(ValueError):
            build_consensus(judgment(3, 2), judgment(3, 2), judgment(2, 2))

    def test_one_axis_disagreement_uses_axis_median(self):
        result = build_consensus(judgment(3, 2), judgment(2, 2), judgment(2, 2))
        self.assertEqual((result["factuality"], result["mechanism"]), (2, 2))
        self.assertTrue(result["accuracy_pass_v2"])
        self.assertTrue(result["tiebreaker_used"])

    def test_two_axis_disagreement_uses_each_axis_median(self):
        result = build_consensus(judgment(3, 2), judgment(1, 0), judgment(2, 1))
        self.assertEqual((result["factuality"], result["mechanism"]), (2, 1))
        self.assertFalse(result["accuracy_pass_v2"])


class AccuracyV2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.judges = {
            "primary": ["openai-group/gpt-5.4", "claude-group/claude-opus-4-7"],
            "tiebreaker": "gemini-group/gemini-3.1-pro",
        }

    def test_loader_produces_frozen_96_record_matrix(self):
        records = load_benchmark_records()
        self.assertEqual(len(records), 96)
        self.assertEqual(
            [(record["model_key"], record["concept"]) for record in records[:12]],
            [(MODEL_ORDER[0], concept) for concept in CONCEPTS],
        )
        self.assertEqual(len({(r["model_key"], r["concept"]) for r in records}), 96)

    def test_resume_reuses_matching_hash_and_invalidates_changed_text(self):
        records = load_benchmark_records()[:2]
        first = prepare_output(records, None, self.judges)
        first["records"][0]["judgments"][self.judges["primary"][0]] = judgment()
        first["records"][1]["judgments"][self.judges["primary"][0]] = judgment()

        changed = [dict(records[0]), dict(records[1])]
        changed[0]["text"] += " changed"
        resumed = prepare_output(changed, first, self.judges)
        self.assertEqual(resumed["records"][0]["judgments"], {})
        self.assertIn(self.judges["primary"][0], resumed["records"][1]["judgments"])

    def test_gemini_task_created_only_for_axis_disagreement(self):
        data = prepare_output(load_benchmark_records()[:2], None, self.judges)
        first, second = self.judges["primary"]
        data["records"][0]["judgments"] = {first: judgment(3, 2), second: judgment(3, 2)}
        data["records"][1]["judgments"] = {first: judgment(3, 2), second: judgment(2, 2)}
        tasks = tiebreaker_tasks(data, self.judges["primary"], self.judges["tiebreaker"])
        self.assertEqual(tasks, [(1, self.judges["tiebreaker"])])

    def test_invalid_response_retries_then_accepts_valid_json(self):
        invalid = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "factuality": 4,
                "mechanism": 2,
                "errors": [],
                "justification": "invalid range",
            })))]
        )
        valid = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "factuality": 3,
                "mechanism": 2,
                "errors": [],
                "justification": "valid response",
            })))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(side_effect=[invalid, valid]))
            )
        )
        record = load_benchmark_records()[0]
        with patch("litmus.judge_accuracy_v2.time.sleep"):
            result = judge_one(client, self.judges["primary"][0], record, max_retries=2)
        self.assertEqual((result["factuality"], result["mechanism"]), (3, 2))
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_unresolved_primary_or_tiebreaker_stops_finalization(self):
        data = prepare_output(load_benchmark_records()[:1], None, self.judges)
        with self.assertRaises(RuntimeError):
            finalize_consensus(data, self.judges["primary"], self.judges["tiebreaker"])

        first, second = self.judges["primary"]
        data["records"][0]["judgments"] = {
            first: judgment(3, 2),
            second: judgment(2, 2),
        }
        with self.assertRaises(RuntimeError):
            finalize_consensus(data, self.judges["primary"], self.judges["tiebreaker"])

    def test_preflight_fails_when_required_model_is_missing(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=Mock(return_value=SimpleNamespace(data=[SimpleNamespace(id="only-one")]))
            )
        )
        with self.assertRaises(RuntimeError):
            preflight_models(client, ["only-one", "missing"])

    def test_custom_input_loader_accepts_sweep_metadata(self):
        record = dict(load_benchmark_records()[0])
        record["temperature_seed"] = "t0p3_s1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"records": [record]}), encoding="utf-8")
            loaded = load_input_records(path)
        self.assertEqual(loaded[0]["temperature_seed"], "t0p3_s1")

    def test_custom_input_loader_rejects_duplicate_identity(self):
        record = load_benchmark_records()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps({"records": [record, record]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_input_records(path)

    def test_versioned_report_keeps_v2_first_and_history_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text("# Report\n\n| old | table |\n|---|---|\n", encoding="utf-8")
            install_versioned_report(path, "## Accuracy-v2\n\n| new | table |")
            install_versioned_report(path, "## Accuracy-v2\n\n| newer | table |")
            text = path.read_text(encoding="utf-8")
            self.assertLess(text.index(V2_START), text.index(HISTORICAL_START))
            self.assertEqual(text.count(V2_START), 1)
            self.assertEqual(text.count(HISTORICAL_START), 1)
            self.assertIn("| old | table |", text)
            self.assertIn("| newer | table |", text)

    def test_aggregate_reports_all_eight_models(self):
        data = prepare_output(load_benchmark_records(), None, self.judges)
        first, second = self.judges["primary"]
        for record in data["records"]:
            record["judgments"] = {first: judgment(3, 2), second: judgment(3, 2)}
            record["consensus"] = build_consensus(
                record["judgments"][first], record["judgments"][second]
            )
            record["overall_pass_v2"] = bool(record["readability_pass"])
        results = aggregate(data)
        self.assertEqual(results["model_order"], MODEL_ORDER)
        self.assertEqual(len(results["models"]), 8)
        self.assertEqual(results["agreement"]["exact_axis_agreement"], 96)

    def test_sweep_helpers_rank_readability_distance(self):
        self.assertEqual(setting_key(0.3, 2), "t0p3_s2")
        self.assertEqual(
            readability_penalty(
                {
                    "whole_passage_fk": 4.0,
                    "whole_passage_ari": 5.0,
                    "fk_stdev": 1.0,
                }
            ),
            0.0,
        )
        self.assertGreater(
            readability_penalty(
                {
                    "whole_passage_fk": 2.0,
                    "whole_passage_ari": 2.0,
                    "fk_stdev": 2.0,
                }
            ),
            0.0,
        )

    def test_sweep_summary_orders_best_overall_first(self):
        first = dict(load_benchmark_records()[0])
        second = dict(load_benchmark_records()[1])
        for record, model_key, passed in (
            (first, "weak", False),
            (second, "strong", True),
        ):
            record["model_key"] = model_key
            record["label"] = model_key
            record["consensus"] = {
                "clean_pass": passed,
                "accuracy_pass_v2": passed,
                "tiebreaker_used": False,
            }
            record["readability_pass"] = passed
            record["overall_pass_v2"] = passed
        rows = summarize({"records": [first, second]})
        self.assertEqual(rows[0]["model_key"], "strong")


if __name__ == "__main__":
    unittest.main()
