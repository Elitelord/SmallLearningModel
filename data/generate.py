"""Part A.3-A.5 - the generate -> readability-rewrite -> accuracy-gate pipeline.

For each TRAIN concept:
  1. Teacher writes a candidate explanation, few-shot-anchored on data/exemplars.json.
  2. Score every sentence with the litmus FK harness (score_text).
  3. If any sentence > FK 3.0 or <70% in band: send the offending sentences and
     their scores back to the teacher - "these are too hard, rewrite them simpler,
     keep the science correct." Repeat up to --max-rewrites times.
  4. If it still fails after the cap: DISCARD (don't force it).
  5. Accuracy gate: a judge (NOT the student, and a different model from the
     teacher) scores 0/1/2. KEEP ONLY 2s. A wrong example is poison for a 0.6B
     student, so the accuracy bar is strict.

Yield is logged at every stage (readability discards, accuracy discards, avg
rewrite iterations) - this is the DOK evidence for whether the pipeline is viable.

Writes data/generated_v0.jsonl - one JSON object per KEPT example:
    {"concept", "explanation", "fk", "accuracy", "rewrite_iters"}

Modes:
    # full pipeline, real dataset
    .venv\\Scripts\\python -m data.generate
    # SMOKE / Part E: ~50 quick "junk" examples, skip heavy filtering
    .venv\\Scripts\\python -m data.generate --junk --limit 50 --out data/generated_v0.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from litmus.accuracy import build_judge_prompt
from litmus.env import load_env
from litmus.fk_score import BAND, CEILING, score_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
CONCEPTS_PATH = HERE / "concepts.json"
EXEMPLARS_PATH = HERE / "exemplars.json"

GEN_SYSTEM = (
    "You write science explanations for a 7-year-old in 3rd grade. Follow these rules:\n"
    "- Every sentence must be simple enough for an 8-year-old to read alone. Short "
    "sentences, common everyday words, mostly one- or two-syllable words.\n"
    "- Avoid technical terms; if one is unavoidable, explain it in plain words at once.\n"
    "- Be scientifically ACCURATE and explain the real HOW/WHY (the mechanism), not "
    "just a definition. Never oversimplify into something wrong.\n"
    "- Write 4 to 6 sentences. Return only the explanation text, no preamble."
)


def few_shot_block(exemplars: list[dict]) -> str:
    lines = ["Here are examples of the target style:\n"]
    for e in exemplars:
        lines.append(f"Concept: {e['concept']}\nExplanation: {e['explanation']}\n")
    return "\n".join(lines)


def fk_feedback(concept: str, text: str, score: dict) -> str:
    """Build the rewrite instruction listing the sentences that are too hard."""
    over = [r for r in score["sentences"] if r["over_ceiling"]]
    bits = [
        f'These sentences read too hard (Flesch-Kincaid grade must stay <= {CEILING}, '
        f"and at least 70% of sentences should sit in {BAND[0]}-{BAND[1]}):"
    ]
    for r in over:
        bits.append(f'  - FK {r["fk"]}: "{r["sentence"]}"')
    if not over:
        bits.append(
            f'  (No single sentence is over {CEILING}, but only {int(score["pct_in_band"]*100)}% '
            f"of sentences are in the {BAND[0]}-{BAND[1]} band; too many are trivially short.)"
        )
    bits.append(
        "\nRewrite the WHOLE explanation. Make the flagged sentences simpler with "
        "shorter, more common words, but keep every science fact correct and keep the "
        "mechanism. Return only the new explanation."
    )
    return "\n".join(bits)


def generate_candidate(client, model, concept, fewshot, temperature):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"{fewshot}\nNow do this one.\nConcept: {concept}\nExplanation:"},
        ],
    )
    return resp.choices[0].message.content.strip()


def rewrite_candidate(client, model, concept, prev, feedback, fewshot, temperature):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"{fewshot}\nConcept: {concept}\nExplanation:"},
            {"role": "assistant", "content": prev},
            {"role": "user", "content": feedback},
        ],
    )
    return resp.choices[0].message.content.strip()


def judge_accuracy(client, judge_model, concept, text):
    """0/1/2 mechanism-rubric judge. Judge != student; different model from teacher."""
    resp = client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": build_judge_prompt(concept, text)}],
    )
    data = json.loads(resp.choices[0].message.content)
    return int(data["score"]), data.get("justification", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="gpt-4o-mini", help="generation/rewrite model")
    ap.add_argument("--judge", default="gpt-4o", help="accuracy judge (!= student, != teacher)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-rewrites", type=int, default=3, help="readability rewrite cap")
    ap.add_argument("--limit", type=int, default=None, help="cap number of concepts")
    ap.add_argument("--out", default=str(HERE / "generated_v0.jsonl"))
    ap.add_argument(
        "--junk",
        action="store_true",
        help="SMOKE mode: one candidate per concept, NO rewrite loop, NO accuracy gate. "
        "Produces throwaway data fast to exercise the train/eval plumbing (Part E).",
    )
    args = ap.parse_args()

    concepts_data = json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))
    train_concepts = concepts_data["train"]
    if args.limit:
        train_concepts = train_concepts[: args.limit]
    exemplars = json.loads(EXEMPLARS_PATH.read_text(encoding="utf-8"))["exemplars"]
    fewshot = few_shot_block(exemplars)

    load_env()
    from openai import OpenAI

    client = OpenAI()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "n_concepts": len(train_concepts),
        "kept": 0,
        "discarded_readability": 0,
        "discarded_accuracy": 0,
        "accuracy_hist": {0: 0, 1: 0, 2: 0},
        "rewrite_iters_total": 0,
        "readability_passers": 0,
    }

    mode = "JUNK (no filtering)" if args.junk else "FULL pipeline"
    print(f"=== data-gen: {mode} ===")
    print(f"teacher={args.teacher}  judge={args.judge}  concepts={len(train_concepts)}  "
          f"max_rewrites={args.max_rewrites}\n")

    kept_records = []
    for i, concept in enumerate(train_concepts, 1):
        text = generate_candidate(client, args.teacher, concept, fewshot, args.temperature)
        score = score_text(text)

        if args.junk:
            # Skip both gates - throwaway data to test plumbing only.
            rec = {
                "concept": concept,
                "explanation": text,
                "fk": {"max_fk": score.get("max_fk"), "pct_in_band": score.get("pct_in_band"),
                       "readability_pass": score.get("readability_pass")},
                "accuracy": {"score": None, "note": "junk-mode: accuracy gate skipped"},
                "rewrite_iters": 0,
            }
            kept_records.append(rec)
            stats["kept"] += 1
            print(f"[{i}/{len(train_concepts)}] JUNK kept | {concept}")
            continue

        # --- readability rewrite loop ---
        iters = 0
        while not score.get("readability_pass") and iters < args.max_rewrites:
            iters += 1
            feedback = fk_feedback(concept, text, score)
            text = rewrite_candidate(client, args.teacher, concept, text, feedback,
                                     fewshot, args.temperature)
            score = score_text(text)
        stats["rewrite_iters_total"] += iters

        if not score.get("readability_pass"):
            stats["discarded_readability"] += 1
            print(f"[{i}/{len(train_concepts)}] DISCARD readability (max_fk="
                  f"{score.get('max_fk')}, band={score.get('pct_in_band')}, {iters} rewrites) | {concept}")
            continue
        stats["readability_passers"] += 1

        # --- accuracy gate: keep only 2s ---
        acc_score, note = judge_accuracy(client, args.judge, concept, text)
        stats["accuracy_hist"][acc_score] = stats["accuracy_hist"].get(acc_score, 0) + 1
        if acc_score != 2:
            stats["discarded_accuracy"] += 1
            print(f"[{i}/{len(train_concepts)}] DISCARD accuracy={acc_score} ({iters} rewrites) | {concept}")
            continue

        rec = {
            "concept": concept,
            "explanation": text,
            "fk": {"max_fk": score["max_fk"], "pct_in_band": score["pct_in_band"],
                   "readability_pass": True},
            "accuracy": {"score": acc_score, "note": note},
            "rewrite_iters": iters,
        }
        kept_records.append(rec)
        stats["kept"] += 1
        print(f"[{i}/{len(train_concepts)}] KEEP (acc=2, {iters} rewrites, "
              f"max_fk={score['max_fk']}) | {concept}")

    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- yield report ---
    n = stats["n_concepts"]
    avg_iters = stats["rewrite_iters_total"] / n if n else 0
    print("\n=== YIELD REPORT ===")
    print(f"concepts attempted:       {n}")
    print(f"discarded (readability):  {stats['discarded_readability']}")
    print(f"passed readability:       {stats['readability_passers']}")
    print(f"discarded (accuracy != 2):{stats['discarded_accuracy']}")
    print(f"accuracy histogram:       {stats['accuracy_hist']}")
    print(f"KEPT (final dataset):     {stats['kept']}  ({stats['kept']/n*100:.0f}% yield)" if n else "KEPT: 0")
    print(f"avg rewrite iterations:   {avg_iters:.2f}")
    print(f"\nWrote {out_path} ({stats['kept']} examples)")

    stats_path = out_path.with_suffix(".stats.json")
    stats["avg_rewrite_iters"] = round(avg_iters, 2)
    stats["mode"] = "junk" if args.junk else "full"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
