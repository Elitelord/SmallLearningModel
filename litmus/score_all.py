"""Score every (model, concept), then emit results.json + a markdown table.

Combines:
  - API/local outputs written to litmus/outputs/*.json by run_api / run_qwen
  - manually pasted browser outputs in litmus/manual_outputs.json
  - accuracy judgments in litmus/accuracy_scores.json

overall_pass = readability_pass AND accuracy_pass (== accuracy score of 2).

Usage:
    .venv\\Scripts\\python -m litmus.score_all
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from litmus.accuracy import accuracy_pass, load_accuracy
from litmus.concepts import CONCEPTS
from litmus.fk_score import BAND_MIN_PASS, CEILING, score_text

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
MANUAL_PATH = HERE / "manual_outputs.json"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"

# Preferred display order + friendly labels.
MODEL_LABELS = {
    "gpt": "GPT (gpt-4o, API)",
    "gemini": "Gemini (browser)",
    "claude_browser": "Claude (browser)",
    "qwen_0.6b": "Qwen3-0.6B (local)",
    "qwen_1.7b": "Qwen3-1.7B (local)",
}


def load_all_outputs() -> dict:
    """Return {model_key: {"model_id", "temperature", "outputs": {concept: text}}}."""
    models = {}
    # Structured outputs from the runners.
    if OUT_DIR.exists():
        for p in sorted(OUT_DIR.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            models[d["model_key"]] = {
                "model_id": d.get("model_id", d["model_key"]),
                "temperature": d.get("temperature", "n/a"),
                "outputs": d["outputs"],
            }
    # Manually pasted browser outputs (temps not controllable).
    if MANUAL_PATH.exists():
        manual = json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
        for key, outputs in manual.items():
            outputs = {c: t for c, t in outputs.items() if t and t.strip()}
            if not outputs:
                continue  # skip empty scaffold entries
            models.setdefault(
                key, {"model_id": "browser (manual paste)", "temperature": "uncontrolled", "outputs": {}}
            )
            models[key]["outputs"].update(outputs)
    return models


def failure_mode(fk: dict, acc_score, readability_pass: bool) -> str:
    """One-line human-readable failure note (or 'pass')."""
    if not readability_pass:
        if fk["n_over_ceiling"] > 0:
            # locate the first offending sentence for a concrete note
            idx = next(i for i, r in enumerate(fk["sentences"], 1) if r["over_ceiling"])
            return (
                f"sentence {idx} hits FK {fk['sentences'][idx-1]['fk']} "
                f"(> {CEILING} ceiling); {fk['n_over_ceiling']} over, max {fk['max_fk']}"
            )
        return f"only {int(fk['pct_in_band']*100)}% of sentences in band (need {int(BAND_MIN_PASS*100)}%)"
    # readability passed — is accuracy the problem?
    if acc_score is None:
        return "reading level OK; accuracy UNSCORED"
    if acc_score == 1:
        return "reading level OK but surface definition, no mechanism (acc=1)"
    if acc_score == 0:
        return "reading level OK but factual error / oversimplification (acc=0)"
    return "pass"


def dominant_failure(rows: list) -> str:
    cats = []
    for r in rows:
        if r["overall_pass"]:
            continue
        note = r["failure_note"]
        if "ceiling" in note:
            cats.append("sentence(s) over FK 3.0 ceiling")
        elif "in band" in note:
            cats.append("too few sentences in the 2.0-3.0 band")
        elif "no mechanism" in note:
            cats.append("grade OK but skips the mechanism (acc=1)")
        elif "factual error" in note:
            cats.append("grade OK but factually wrong (acc=0)")
        elif "UNSCORED" in note:
            cats.append("accuracy unscored")
        else:
            cats.append("other")
    if not cats:
        return "none — passed all scored concepts"
    return Counter(cats).most_common(1)[0][0]


def main():
    models = load_all_outputs()
    accuracy = load_accuracy()

    results = {}
    for model_key, info in models.items():
        rows = []
        for concept in CONCEPTS:
            text = info["outputs"].get(concept)
            if not text or not text.strip():
                continue  # this model has no output for this concept yet
            fk = score_text(text)
            if "error" in fk:
                continue  # degenerate output (no parseable sentences)
            acc = accuracy.get(model_key, {}).get(concept, {})
            acc_score = acc.get("score")
            acc_note = acc.get("note", "")
            r_pass = fk.get("readability_pass", False)
            a_pass = accuracy_pass(acc_score)
            row = {
                "concept": concept,
                "max_fk": fk.get("max_fk"),
                "n_over_ceiling": fk.get("n_over_ceiling"),
                "pct_in_band": fk.get("pct_in_band"),
                "n_short_flag": fk.get("n_short_flag"),
                "readability_pass": r_pass,
                "accuracy_score": acc_score,
                "accuracy_note": acc_note,
                "overall_pass": bool(r_pass and a_pass),
            }
            row["failure_note"] = failure_mode(fk, acc_score, r_pass)
            rows.append(row)

        n = len(rows)
        n_read = sum(1 for r in rows if r["readability_pass"])
        n_acc = sum(1 for r in rows if r["accuracy_score"] == 2)
        n_overall = sum(1 for r in rows if r["overall_pass"])
        n_unscored = sum(1 for r in rows if r["accuracy_score"] is None)
        results[model_key] = {
            "label": MODEL_LABELS.get(model_key, model_key),
            "model_id": info["model_id"],
            "temperature": info["temperature"],
            "n_concepts": n,
            "n_readability_pass": n_read,
            "n_accuracy_pass": n_acc,
            "n_overall_pass": n_overall,
            "n_accuracy_unscored": n_unscored,
            "dominant_failure": dominant_failure(rows),
            "concepts": rows,
        }

    RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(results)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {RESULTS_JSON}\nWrote {RESULTS_MD}")


def render_markdown(results: dict) -> str:
    lines = ["# Litmus results\n"]
    lines.append(
        "Behavior Spec: no sentence > FK 3.0; ≥70% of sentences in FK 2.0–3.0; "
        "factually correct AND conveys the mechanism.\n"
    )
    lines.append("`overall_pass = readability_pass AND accuracy_pass (score == 2)`\n")

    # Headline per model.
    lines.append("## Headline\n")
    order = list(MODEL_LABELS.keys())
    ordered_keys = [k for k in order if k in results] + [
        k for k in results if k not in order
    ]
    for k in ordered_keys:
        r = results[k]
        unscored = (
            f" ({r['n_accuracy_unscored']} concept(s) accuracy-unscored)"
            if r["n_accuracy_unscored"]
            else ""
        )
        lines.append(
            f"- **{r['label']}** passed the full spec on "
            f"**{r['n_overall_pass']}/{r['n_concepts']}** concepts"
            f" (readability {r['n_readability_pass']}/{r['n_concepts']}, "
            f"accuracy=2 on {r['n_accuracy_pass']}/{r['n_concepts']}). "
            f"Dominant failure mode: {r['dominant_failure']}.{unscored}"
        )
    lines.append("")

    # Per-model detail tables.
    for k in ordered_keys:
        r = results[k]
        lines.append(
            f"## {r['label']}  —  model_id=`{r['model_id']}`, temp={r['temperature']}\n"
        )
        lines.append(
            "| Concept | max_fk | over | %band | read | acc | overall | failure note |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in r["concepts"]:
            acc = "–" if c["accuracy_score"] is None else c["accuracy_score"]
            lines.append(
                f"| {c['concept']} | {c['max_fk']} | {c['n_over_ceiling']} | "
                f"{int(c['pct_in_band']*100)}% | "
                f"{'✅' if c['readability_pass'] else '❌'} | {acc} | "
                f"{'✅' if c['overall_pass'] else '❌'} | {c['failure_note']} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
