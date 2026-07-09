"""Re-run the ORIGINAL litmus test under the Day-4 v4 gate (FK 3-6 AND ARI 3-7).

Same 12 elementary-science concepts, same saved model outputs, same accuracy
judgments (reused from results_v3.json -- gpt-4o audience-calibrated, no re-judge).
The ONLY thing that changes vs results_v3 is the readability gate: the v3 band was
whole-passage FK 1.5-3.0 (shown to target ~grade 1-2, see eval/metric_comparison_
real.md); v4 uses the recalibrated grade-3 band FK 3.0-6.0 AND the co-best metric
ARI 3.0-7.0.

Models (the 4 the user asked for, + 0.6B as the small-capacity reference):
    GPT (gpt-4o), Claude (browser), Gemini (browser), Qwen3-4B (local, instruct-style).

overall_pass_v4 = readability_pass_v4 AND accuracy==2.

Usage: .venv\\Scripts\\python -m litmus.rescore_v4
Writes litmus/results_v4.json + litmus/results_v4.md.
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from litmus.concepts import CONCEPTS
from litmus.fk_score import (ARI_BAND_V4, DISPERSION_MAX_V4, WP_BAND_V4,
                             score_text)
from litmus.score_all import load_all_outputs

HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "results_v3.json"
OUT_JSON = HERE / "results_v4.json"
OUT_MD = HERE / "results_v4.md"

# the 4 requested models, in display order (+ 0.6B reference at the end)
ORDER = ["gpt", "claude_browser", "gemini", "qwen_4b", "qwen_0.6b"]
LABELS = {
    "gpt": "GPT (gpt-4o)",
    "claude_browser": "Claude (browser)",
    "gemini": "Gemini (browser)",
    "qwen_4b": "Qwen3-4B (local, instruct-style)",
    "qwen_0.6b": "Qwen3-0.6B (local, reference)",
}


def load_accuracy_from_v3() -> dict:
    """{model_key: {concept: accuracy_score}} reused from the v3 run (no re-judge)."""
    if not V3_PATH.exists():
        return {}
    v3 = json.loads(V3_PATH.read_text(encoding="utf-8"))
    acc = {}
    for mkey, mdata in v3.items():
        acc[mkey] = {c["concept"]: c.get("accuracy") for c in mdata.get("concepts", [])}
    return acc


def main():
    models = load_all_outputs()
    accuracy = load_accuracy_from_v3()

    results = {}
    for mkey in [k for k in ORDER if k in models] + [k for k in models if k not in ORDER]:
        info = models[mkey]
        rows = []
        for concept in CONCEPTS:
            text = info["outputs"].get(concept)
            if not text or not text.strip():
                continue
            fk = score_text(text)
            if "error" in fk:
                continue
            acc_score = accuracy.get(mkey, {}).get(concept)
            r_pass = fk["readability_pass_v4"]
            rows.append({
                "concept": concept,
                "whole_passage_fk": fk["whole_passage_fk"],
                "whole_passage_ari": fk["whole_passage_ari"],
                "fk_stdev": fk["fk_stdev"],
                "cond_fk": fk["cond_wp_band_v4"],
                "cond_ari": fk["cond_ari_band_v4"],
                "cond_disp": fk["cond_dispersion_v4"],
                "cond_backstop": fk["cond_backstop_v4"],
                "readability_pass_v4": r_pass,
                "accuracy": acc_score,
                "overall_pass": bool(r_pass and acc_score == 2),
            })
        n = len(rows)
        results[mkey] = {
            "label": LABELS.get(mkey, mkey),
            "model_id": info["model_id"],
            "n": n,
            "n_readability_v4": sum(1 for r in rows if r["readability_pass_v4"]),
            "n_accuracy_2": sum(1 for r in rows if r["accuracy"] == 2),
            "n_overall_pass": sum(1 for r in rows if r["overall_pass"]),
            "concepts": rows,
        }

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_md(results)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


def render_md(results: dict) -> str:
    L = [f"# Litmus baseline under the v4 gate (FK {WP_BAND_V4[0]}-{WP_BAND_V4[1]} "
         f"AND ARI {ARI_BAND_V4[0]}-{ARI_BAND_V4[1]}, dispersion ≤ {DISPERSION_MAX_V4})\n",
         "Same 12 concepts, same saved outputs, same accuracy judgments as `results_v3.md` "
         "(gpt-4o, audience-calibrated). Only the readability gate changed: v3's FK 1.5-3.0 "
         "band was shown to target ~grade 1-2 (`eval/metric_comparison_real.md`); v4 uses the "
         "recalibrated real-grade-3 band FK 3-6 plus the co-best co-metric ARI 3-7.\n",
         "`overall_pass = readability_pass_v4 AND accuracy==2`\n",
         "## Headline (v4)\n",
         "| Model | readability (v4) | accuracy=2 | overall pass |",
         "|---|---|---|---|"]
    for k in [x for x in ORDER if x in results]:
        r = results[k]
        L.append(f"| {r['label']} | {r['n_readability_v4']}/{r['n']} | "
                 f"{r['n_accuracy_2']}/{r['n']} | **{r['n_overall_pass']}/{r['n']}** |")
    L.append("")
    for k in [x for x in ORDER if x in results]:
        r = results[k]
        L.append(f"## {r['label']}  —  `{r['model_id']}`\n")
        L.append("| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in r["concepts"]:
            L.append(f"| {c['concept']} | {c['whole_passage_fk']} | {c['whole_passage_ari']} | "
                     f"{c['fk_stdev']} | {'✅' if c['cond_fk'] else '❌'} | "
                     f"{'✅' if c['cond_ari'] else '❌'} | "
                     f"{'✅' if c['readability_pass_v4'] else '❌'} | "
                     f"{c['accuracy']} | {'✅' if c['overall_pass'] else '❌'} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
