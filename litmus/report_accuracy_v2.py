"""Aggregate accuracy-v2 judgments and install versioned Markdown reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from litmus.accuracy_v2 import (
    FACTUALITY_MAX,
    MECHANISM_MAX,
    RUBRIC_VERSION,
    SCHEMA_VERSION,
    build_consensus,
    judges_disagree,
    linear_weighted_kappa,
)
from litmus.benchmark_v2 import MODEL_ORDER

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORES = REPO_ROOT / "litmus" / "accuracy_v2_scores.json"
DEFAULT_RESULTS = REPO_ROOT / "litmus" / "results_v4_accuracy_v2.json"
MODEL_COMPARISON = REPO_ROOT / "eval" / "model_comparison.md"
LITMUS_RESULTS = REPO_ROOT / "litmus" / "results_v4.md"

V2_START = "<!-- accuracy-v2:start -->"
V2_END = "<!-- accuracy-v2:end -->"
HISTORICAL_START = "<!-- accuracy-v1-historical:start -->"
HISTORICAL_END = "<!-- accuracy-v1-historical:end -->"


def _round(value: float) -> float:
    return round(value, 3)


def _escape(value: str, limit: int = 220) -> str:
    cleaned = " ".join(value.replace("|", "/").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _score_pair(judgment: dict | None) -> str:
    return "—" if judgment is None else f"{judgment['factuality']}/{judgment['mechanism']}"


def _consensus_note(record: dict, judge_order: list[str]) -> str:
    consensus = record["consensus"]
    selected = None
    for model in reversed(judge_order):
        judgment = record["judgments"].get(model)
        if not judgment:
            continue
        if (
            judgment["factuality"] == consensus["factuality"]
            and judgment["mechanism"] == consensus["mechanism"]
        ):
            selected = judgment
            break
    if selected is None:
        selected = record["judgments"][judge_order[0]]
    if selected["errors"]:
        error = selected["errors"][0]
        return _escape(
            f"{error['severity']}: {error['claim']} Correction: {error['correction']}"
        )
    return _escape(selected["justification"])


def aggregate(scores: dict) -> dict:
    if scores.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported accuracy-v2 score schema")
    if scores.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError("unexpected rubric version")

    records = scores.get("records", [])
    expected = len(MODEL_ORDER) * 12
    if len(records) != expected:
        raise ValueError(f"expected {expected} complete records, found {len(records)}")
    if any(record.get("consensus") is None for record in records):
        raise ValueError("cannot report unresolved judgments")

    primary = scores["judges"]["primary"]
    tiebreaker = scores["judges"]["tiebreaker"]
    for record in records:
        first = record["judgments"].get(primary[0])
        second = record["judgments"].get(primary[1])
        if first is None or second is None:
            raise ValueError("every record requires both primary judgments")
        disagreement = judges_disagree(first, second)
        third = record["judgments"].get(tiebreaker)
        if disagreement != (third is not None):
            raise ValueError("Gemini must appear if and only if primary axes disagree")
        expected_consensus = build_consensus(first, second, third)
        if record["consensus"] != expected_consensus:
            raise ValueError("stored consensus does not match the judge scores")
        expected_overall = bool(
            record["readability_pass"] and expected_consensus["accuracy_pass_v2"]
        )
        if record.get("overall_pass_v2") != expected_overall:
            raise ValueError("stored overall_pass_v2 is inconsistent")

    by_model = defaultdict(list)
    for record in records:
        by_model[record["model_key"]].append(record)

    models = {}
    for model_key in MODEL_ORDER:
        rows = by_model.get(model_key, [])
        if len(rows) != 12:
            raise ValueError(f"expected 12 rows for {model_key}, found {len(rows)}")
        models[model_key] = {
            "label": rows[0]["label"],
            "model_id": rows[0]["model_id"],
            "prompt": rows[0]["prompt"],
            "tested_family": rows[0]["tested_family"],
            "n": len(rows),
            "n_readability_pass": sum(row["readability_pass"] for row in rows),
            "n_clean_pass": sum(row["consensus"]["clean_pass"] for row in rows),
            "n_accuracy_pass_v2": sum(
                row["consensus"]["accuracy_pass_v2"] for row in rows
            ),
            "n_overall_pass_v2": sum(row["overall_pass_v2"] for row in rows),
            "mean_factuality": _round(
                sum(row["consensus"]["factuality"] for row in rows) / len(rows)
            ),
            "mean_mechanism": _round(
                sum(row["consensus"]["mechanism"] for row in rows) / len(rows)
            ),
            "n_primary_disagreement": sum(
                row["consensus"]["primary_disagreement"] for row in rows
            ),
            "n_tiebreaker_used": sum(
                row["consensus"]["tiebreaker_used"] for row in rows
            ),
            "concepts": rows,
        }

    first_factuality = [record["judgments"][primary[0]]["factuality"] for record in records]
    second_factuality = [record["judgments"][primary[1]]["factuality"] for record in records]
    first_mechanism = [record["judgments"][primary[0]]["mechanism"] for record in records]
    second_mechanism = [record["judgments"][primary[1]]["mechanism"] for record in records]
    exact = sum(
        first_factuality[index] == second_factuality[index]
        and first_mechanism[index] == second_mechanism[index]
        for index in range(len(records))
    )
    pass_agreement = sum(
        (
            first_factuality[index] >= 2 and first_mechanism[index] == 2
        )
        == (
            second_factuality[index] >= 2 and second_mechanism[index] == 2
        )
        for index in range(len(records))
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judges": scores["judges"],
        "agreement": {
            "n": len(records),
            "exact_axis_agreement": exact,
            "exact_axis_agreement_rate": _round(exact / len(records)),
            "accuracy_pass_agreement": pass_agreement,
            "accuracy_pass_agreement_rate": _round(pass_agreement / len(records)),
            "factuality_linear_weighted_kappa": _round(
                linear_weighted_kappa(first_factuality, second_factuality, FACTUALITY_MAX)
            ),
            "mechanism_linear_weighted_kappa": _round(
                linear_weighted_kappa(first_mechanism, second_mechanism, MECHANISM_MAX)
            ),
            "gemini_tiebreakers": sum(
                record["consensus"]["tiebreaker_used"] for record in records
            ),
        },
        "model_order": MODEL_ORDER,
        "models": models,
        "tiebreaker": tiebreaker,
    }


def render_summary(results: dict) -> str:
    primary = results["judges"]["primary"]
    tiebreaker = results["judges"]["tiebreaker"]
    lines = [
        "## Accuracy-v2 Multi-Judge Results",
        "",
        f"Rubric `{RUBRIC_VERSION}` uses factuality 0–3 and mechanism 0–2. "
        "A clean pass is 3/2; the benchmark accuracy pass allows a minor localized "
        "error (factuality ≥2) but still requires mechanism 2. Overall pass also "
        "requires the unchanged v4 readability gate.",
        "",
        f"Primary judges: `{primary[0]}` and `{primary[1]}`. `{tiebreaker}` is called "
        "only when either primary axis differs; consensus is the per-axis median.",
        "",
        "### Headline",
        "",
        "| Model | Prompt | Readability | Clean 3/2 | Accuracy-v2 | Overall-v2 | Mean F/M | Gemini |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key in results["model_order"]:
        model = results["models"][model_key]
        n = model["n"]
        lines.append(
            f"| {model['label']} | {model['prompt']} | {model['n_readability_pass']}/{n} | "
            f"{model['n_clean_pass']}/{n} | {model['n_accuracy_pass_v2']}/{n} | "
            f"**{model['n_overall_pass_v2']}/{n}** | "
            f"{model['mean_factuality']}/{model['mean_mechanism']} | "
            f"{model['n_tiebreaker_used']}/{n} |"
        )

    lines.extend(
        [
            "",
            "### Tuned Iteration Comparison",
            "",
            "| Iteration | Readability | Clean 3/2 | Accuracy-v2 | Overall-v2 | Mean F/M |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_key in ("v4r2", "v4r3", "v4r4"):
        model = results["models"][model_key]
        n = model["n"]
        lines.append(
            f"| {model['label']} | {model['n_readability_pass']}/{n} | "
            f"{model['n_clean_pass']}/{n} | {model['n_accuracy_pass_v2']}/{n} | "
            f"**{model['n_overall_pass_v2']}/{n}** | "
            f"{model['mean_factuality']}/{model['mean_mechanism']} |"
        )

    agreement = results["agreement"]
    lines.extend(
        [
            "",
            "### Judge Agreement",
            "",
            f"- Exact two-axis agreement: {agreement['exact_axis_agreement']}/{agreement['n']} "
            f"({agreement['exact_axis_agreement_rate']:.1%}).",
            f"- Accuracy-pass agreement: {agreement['accuracy_pass_agreement']}/{agreement['n']} "
            f"({agreement['accuracy_pass_agreement_rate']:.1%}).",
            f"- Linear weighted kappa: factuality "
            f"{agreement['factuality_linear_weighted_kappa']}, mechanism "
            f"{agreement['mechanism_linear_weighted_kappa']}.",
            f"- Gemini tiebreakers: {agreement['gemini_tiebreakers']}/{agreement['n']}.",
            "- Judge-family relationships are recorded per output in the raw JSON; "
            "cross-family agreement should be preferred when interpreting tested GPT, Claude, or Gemini rows.",
        ]
    )
    return "\n".join(lines)


def render_detail(results: dict) -> str:
    lines = [render_summary(results), "", "### Per-Concept Detail"]
    judge_order = [*results["judges"]["primary"], results["judges"]["tiebreaker"]]
    for model_key in results["model_order"]:
        model = results["models"][model_key]
        lines.extend(
            [
                "",
                f"#### {model['label']}",
                "",
                "| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for record in model["concepts"]:
            consensus = record["consensus"]
            lines.append(
                f"| {record['concept']} | "
                f"{_score_pair(record['judgments'].get(judge_order[0]))} | "
                f"{_score_pair(record['judgments'].get(judge_order[1]))} | "
                f"{_score_pair(record['judgments'].get(judge_order[2]))} | "
                f"{consensus['factuality']}/{consensus['mechanism']} ({consensus['total']}/5) | "
                f"{'✅' if record['readability_pass'] else '❌'} | "
                f"{'✅' if consensus['accuracy_pass_v2'] else '❌'} | "
                f"{'✅' if record['overall_pass_v2'] else '❌'} | "
                f"{_consensus_note(record, judge_order)} |"
            )
    return "\n".join(lines)


def install_versioned_report(path: Path, accuracy_v2_markdown: str) -> None:
    original = path.read_text(encoding="utf-8")
    block = f"{V2_START}\n{accuracy_v2_markdown.strip()}\n{V2_END}"
    if V2_START in original and V2_END in original:
        before, remainder = original.split(V2_START, 1)
        _, after = remainder.split(V2_END, 1)
        path.write_text(before + block + after, encoding="utf-8")
        return

    lines = original.splitlines()
    title = lines[0] if lines and lines[0].startswith("# ") else f"# {path.stem}"
    historical = "\n".join(lines[1:]).lstrip() if lines else original
    combined = (
        f"{title}\n\n{block}\n\n---\n\n"
        "## Historical Accuracy-v1 Results\n\n"
        "The following tables are preserved from the original 0/1/2 accuracy rubric.\n\n"
        f"{HISTORICAL_START}\n{historical.rstrip()}\n{HISTORICAL_END}\n"
    )
    path.write_text(combined, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--no-docs", action="store_true")
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    results = aggregate(scores)
    args.results.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_docs:
        install_versioned_report(MODEL_COMPARISON, render_summary(results))
        install_versioned_report(LITMUS_RESULTS, render_detail(results))
    print(f"wrote {args.results}")
    if not args.no_docs:
        print(f"updated {MODEL_COMPARISON}")
        print(f"updated {LITMUS_RESULTS}")


if __name__ == "__main__":
    main()
