"""Part A.3-A.5 - the generate -> readability-rewrite -> accuracy-gate pipeline.

For each TRAIN (concept, phrasing) pair:
  1. Teacher writes a candidate explanation, few-shot-anchored on data/exemplars.json.
  2. Score every sentence with the litmus FK harness (score_text).
  3. If it fails the readability spec, send DIRECTION-AWARE feedback back to the
     teacher - flagged sentences that are too HARD (FK > 3.0) get "make simpler",
     flagged sentences that are too EASY (FK < 2.0, dragging the band down) get
     "combine/enrich". Repeat up to --max-rewrites.  [Day-2 bug fixed: the old
     feedback always said "make simpler", which pushed already-too-easy text
     further below the 2.0 band and yielded 0% survival.]
  4. If it still fails after the cap: DISCARD (don't force it).
  5. Length/format gate: enforce the 4-8 sentence shape; drop vacuous/runaway.
  6. Accuracy gate: a judge (NOT the student, different model from the teacher)
     scores 0/1/2. KEEP ONLY 2s. A wrong example is poison for a 0.6B student.

Yield is logged at every stage. Writes JSONL - one JSON object per KEPT example:
    {"concept", "phrasing", "explanation", "fk", "accuracy", "rewrite_iters", "n_sentences"}

Modes:
    # full pipeline, real dataset
    .venv\\Scripts\\python -m data.generate --out data/v1/generated.jsonl
    # ~40-example REVIEW SAMPLE (Step 2 gate), varied concepts + phrasings
    .venv\\Scripts\\python -m data.generate --sample 40 --out data/sample/review_sample.jsonl
    # SMOKE / Part E: quick "junk" examples, skip heavy filtering
    .venv\\Scripts\\python -m data.generate --junk --limit 50 --out data/generated_v0.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from litmus.accuracy import build_judge_prompt
from litmus.env import load_env
from litmus.fk_score import BAND, CEILING, MIN_WORDS, score_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
CONCEPTS_PATH = HERE / "concepts.json"
EXEMPLARS_PATH = HERE / "exemplars.json"

MIN_SENTENCES = 4
MAX_SENTENCES = 8

GEN_SYSTEM = (
    "You write science explanations for a curious 7-year-old in 3rd grade.\n"
    "Follow ALL of these rules:\n"
    "- A child of 8 must be able to read every sentence alone. Use common, everyday "
    "words and mostly one- or two-syllable words.\n"
    "- Aim for sentences of about 8 to 14 words. Do NOT write choppy 3-5 word "
    "sentences (they read as baby-talk) and do NOT write long, winding sentences.\n"
    "- Avoid technical terms; if one is truly needed, explain it in plain words at once.\n"
    "- Be scientifically ACCURATE and explain the real HOW/WHY (the mechanism), not "
    "just a definition. Never oversimplify into something that becomes wrong.\n"
    "- Write 4 to 6 sentences. Return ONLY the explanation text, no preamble, no title."
)


def few_shot_block(exemplars: list[dict]) -> str:
    lines = ["Here are examples of the target style:\n"]
    for e in exemplars:
        lines.append(f"Concept: {e['concept']}\nExplanation: {e['explanation']}\n")
    return "\n".join(lines)


def fk_feedback(text: str, score: dict) -> str:
    """DIRECTION-AWARE rewrite instruction.

    Splits the offending sentences into two buckets and tells the teacher which
    way to move each: too-hard sentences get simplified, too-easy/choppy ones get
    combined or enriched. Fixing only one direction is what tanked Day-2 yield.
    """
    too_hard = [r for r in score["sentences"] if r["over_ceiling"]]
    too_easy = [r for r in score["sentences"]
                if (not r["over_ceiling"]) and (r["fk"] < BAND[0])]

    bits = [
        f"The reading level is off. The spec: NO sentence above Flesch-Kincaid grade "
        f"{CEILING}, and at least 70% of sentences inside grade {BAND[0]}-{BAND[1]}. "
        f"Right now only {int(score['pct_in_band']*100)}% are in that band.",
    ]
    if too_hard:
        bits.append("\nThese sentences are TOO HARD - make them simpler with shorter, "
                     "more common words (or split them):")
        for r in too_hard:
            bits.append(f'  - FK {r["fk"]}: "{r["sentence"]}"')
    if too_easy:
        bits.append("\nThese sentences are TOO SIMPLE / choppy - they drag the level "
                     "below the band. Combine them with a neighbor or add a little "
                     "concrete detail so they read like natural grade-3 sentences "
                     "(about 8-14 words), WITHOUT adding hard words:")
        for r in too_easy:
            flag = " (very short)" if r["short_flag"] else ""
            bits.append(f'  - FK {r["fk"]}{flag}: "{r["sentence"]}"')
    bits.append(
        "\nRewrite the WHOLE explanation, keeping every science fact correct and the "
        "mechanism intact. Stay at 4-6 sentences. Return only the new explanation."
    )
    return "\n".join(bits)


def generate_candidate(client, model, phrasing, fewshot, temperature):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user",
             "content": f"{fewshot}\nNow do this one.\nConcept: {phrasing}\nExplanation:"},
        ],
    )
    return resp.choices[0].message.content.strip()


def rewrite_candidate(client, model, phrasing, prev, feedback, fewshot, temperature):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"{fewshot}\nConcept: {phrasing}\nExplanation:"},
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
        messages=[{"role": "user",
                   "content": build_judge_prompt(concept, text, audience_calibrated=True)}],
    )
    data = json.loads(resp.choices[0].message.content)
    return int(data["score"]), data.get("justification", "")


def build_work_items(cdata: dict, junk: bool):
    """Flatten concepts.json into (concept, phrasing) work items.

    Supports both the scaled schema (train_concepts + phrasings map) and the
    older smoke schema (flat `train` list, no phrasings).
    """
    items = []
    phrasings = cdata.get("phrasings")
    if phrasings:
        for concept, plist in phrasings.items():
            for p in plist:
                items.append({"concept": concept, "phrasing": p})
    else:
        for concept in cdata.get("train_concepts", cdata.get("train", [])):
            items.append({"concept": concept, "phrasing": concept})
    return items


def sample_items(items, k):
    """Deterministic spread-out sample of k items across distinct concepts and
    phrasing positions (no RNG - stays reproducible)."""
    if k >= len(items):
        return items
    # round-robin over concepts so the sample spans many concepts, and walk a
    # stride so we hit different phrasing positions.
    by_concept = {}
    for it in items:
        by_concept.setdefault(it["concept"], []).append(it)
    concepts = list(by_concept)
    out, pos = [], 0
    while len(out) < k:
        c = concepts[pos % len(concepts)]
        bucket = by_concept[c]
        idx = (pos // len(concepts))
        if idx < len(bucket):
            out.append(bucket[idx])
        pos += 1
        if pos > len(items) * 2:
            break
    return out[:k]


def run_authored(client, authored_path, judge_model, out_path):
    """CLAUDE-AS-TEACHER path: score agent-authored explanations through the FK,
    format, and accuracy gates. No API generation - the explanations were written
    (and iterated against score_text) by the agent. The judge still runs."""
    recs = []
    with open(authored_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    stats = {"n_items": len(recs), "kept": 0, "discarded_readability": 0,
             "discarded_format": 0, "discarded_accuracy": 0,
             "accuracy_hist": {0: 0, 1: 0, 2: 0}, "rewrite_iters_total": 0,
             "readability_passers": 0}
    print("=== data-gen: AUTHORED (Claude teacher) ===")
    print(f"teacher=claude(agent)  judge={judge_model}  items={len(recs)}\n")

    kept = []
    for i, rec in enumerate(recs, 1):
        concept = rec["concept"]
        phrasing = rec.get("phrasing", concept)
        text = rec["explanation"].strip()
        score = score_text(text)

        if not score.get("readability_pass_v3"):
            stats["discarded_readability"] += 1
            print(f"[{i}/{len(recs)}] DISCARD readability (wp_fk={score.get('whole_passage_fk')}, "
                  f"band={score.get('pct_in_band')}, long_over={score.get('n_long_over_ceiling')}) "
                  f"| {phrasing}")
            continue
        stats["readability_passers"] += 1

        n_sent = score["n_sentences"]
        if n_sent < MIN_SENTENCES or n_sent > MAX_SENTENCES:
            stats["discarded_format"] += 1
            print(f"[{i}/{len(recs)}] DISCARD format ({n_sent} sentences) | {phrasing}")
            continue

        acc_score, note = judge_accuracy(client, judge_model, phrasing, text)
        stats["accuracy_hist"][acc_score] = stats["accuracy_hist"].get(acc_score, 0) + 1
        if acc_score != 2:
            stats["discarded_accuracy"] += 1
            print(f"[{i}/{len(recs)}] DISCARD accuracy={acc_score} | {phrasing}\n        judge: {note}")
            continue

        kept.append({
            "concept": concept, "phrasing": phrasing, "explanation": text,
            "fk": {"max_fk": score["max_fk"], "pct_in_band": score["pct_in_band"],
                   "whole_passage_fk": score["whole_passage_fk"], "readability_pass_v3": True},
            "accuracy": {"score": acc_score, "note": note},
            "rewrite_iters": rec.get("rewrite_iters", 0), "n_sentences": n_sent,
            "teacher": "claude",
        })
        stats["kept"] += 1
        print(f"[{i}/{len(recs)}] KEEP (acc=2, max_fk={score['max_fk']}, "
              f"band={score['pct_in_band']}, {n_sent} sent) | {phrasing}")

    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = stats["n_items"]
    print("\n=== YIELD REPORT (authored) ===")
    print(f"items:                     {n}")
    print(f"discarded (readability):   {stats['discarded_readability']}")
    print(f"discarded (format 4-8):    {stats['discarded_format']}")
    print(f"discarded (accuracy != 2): {stats['discarded_accuracy']}")
    print(f"accuracy histogram:        {stats['accuracy_hist']}")
    if n:
        print(f"KEPT (final):              {stats['kept']}  ({stats['kept']/n*100:.0f}% yield)")
    print(f"\nWrote {out_path} ({stats['kept']} examples)")
    stats_path = out_path.with_suffix(".stats.json")
    stats["yield_rate"] = round(stats["kept"] / n, 3) if n else 0
    stats["mode"] = "authored"
    stats["teacher"] = "claude"
    stats["judge"] = judge_model
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {stats_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="gpt-4o", help="generation/rewrite model (frontier)")
    ap.add_argument("--judge", default="gpt-4o-mini", help="accuracy judge (!= student, != teacher)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-rewrites", type=int, default=4, help="readability rewrite cap")
    ap.add_argument("--limit", type=int, default=None, help="cap number of work items")
    ap.add_argument("--sample", type=int, default=None,
                    help="generate a spread-out review sample of this many items, then stop")
    ap.add_argument("--authored", default=None,
                    help="CLAUDE-AS-TEACHER mode: read a JSONL of {concept,phrasing,explanation} "
                    "authored (and pre-scored against the FK harness) by the agent, and run it "
                    "through the SAME FK + format + accuracy gates. No API generation/rewrite; "
                    "the accuracy judge still runs. Best judge separation (Claude teacher vs GPT judge).")
    ap.add_argument("--out", default=str(HERE / "generated_v0.jsonl"))
    ap.add_argument(
        "--junk",
        action="store_true",
        help="SMOKE mode: one candidate per item, NO rewrite loop, NO gates. "
        "Throwaway data to exercise the train/eval plumbing (Part E).",
    )
    args = ap.parse_args()

    if args.authored:
        load_env()
        from openai import OpenAI

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Claude is the teacher here, so default the judge to the strong, different
        # family (gpt-4o) unless the caller overrode it.
        judge = args.judge if args.judge != ap.get_default("judge") else "gpt-4o"
        run_authored(OpenAI(), args.authored, judge, out_path)
        return

    cdata = json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))
    items = build_work_items(cdata, args.junk)
    if args.sample:
        items = sample_items(items, args.sample)
    if args.limit:
        items = items[: args.limit]
    exemplars = json.loads(EXEMPLARS_PATH.read_text(encoding="utf-8"))["exemplars"]
    fewshot = few_shot_block(exemplars)

    load_env()
    from openai import OpenAI

    client = OpenAI()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "n_items": len(items),
        "kept": 0,
        "discarded_readability": 0,
        "discarded_format": 0,
        "discarded_accuracy": 0,
        "accuracy_hist": {0: 0, 1: 0, 2: 0},
        "rewrite_iters_total": 0,
        "readability_passers": 0,
    }

    mode = "JUNK (no filtering)" if args.junk else ("REVIEW SAMPLE" if args.sample else "FULL pipeline")
    print(f"=== data-gen: {mode} ===")
    print(f"teacher={args.teacher}  judge={args.judge}  items={len(items)}  "
          f"max_rewrites={args.max_rewrites}\n")

    kept_records = []
    for i, item in enumerate(items, 1):
        concept, phrasing = item["concept"], item["phrasing"]
        text = generate_candidate(client, args.teacher, phrasing, fewshot, args.temperature)
        score = score_text(text)

        if args.junk:
            rec = {
                "concept": concept, "phrasing": phrasing, "explanation": text,
                "fk": {"max_fk": score.get("max_fk"), "pct_in_band": score.get("pct_in_band"),
                       "whole_passage_fk": score.get("whole_passage_fk"),
                       "readability_pass_v3": score.get("readability_pass_v3")},
                "accuracy": {"score": None, "note": "junk-mode: accuracy gate skipped"},
                "rewrite_iters": 0, "n_sentences": score.get("n_sentences"),
            }
            kept_records.append(rec)
            stats["kept"] += 1
            print(f"[{i}/{len(items)}] JUNK kept | {phrasing}")
            continue

        # --- readability rewrite loop (direction-aware) ---
        iters = 0
        while not score.get("readability_pass_v3") and iters < args.max_rewrites:
            iters += 1
            feedback = fk_feedback(text, score)
            text = rewrite_candidate(client, args.teacher, phrasing, text, feedback,
                                     fewshot, args.temperature)
            score = score_text(text)
        stats["rewrite_iters_total"] += iters

        if not score.get("readability_pass_v3"):
            stats["discarded_readability"] += 1
            print(f"[{i}/{len(items)}] DISCARD readability (wp_fk={score.get('whole_passage_fk')}, "
                  f"band={score.get('pct_in_band')}, long_over={score.get('n_long_over_ceiling')}, "
                  f"{iters} rw) | {phrasing}")
            continue
        stats["readability_passers"] += 1

        # --- length / format gate ---
        n_sent = score["n_sentences"]
        if n_sent < MIN_SENTENCES or n_sent > MAX_SENTENCES:
            stats["discarded_format"] += 1
            print(f"[{i}/{len(items)}] DISCARD format ({n_sent} sentences) | {phrasing}")
            continue

        # --- accuracy gate: keep only 2s ---
        acc_score, note = judge_accuracy(client, args.judge, phrasing, text)
        stats["accuracy_hist"][acc_score] = stats["accuracy_hist"].get(acc_score, 0) + 1
        if acc_score != 2:
            stats["discarded_accuracy"] += 1
            print(f"[{i}/{len(items)}] DISCARD accuracy={acc_score} ({iters} rw) | {phrasing}")
            continue

        rec = {
            "concept": concept, "phrasing": phrasing, "explanation": text,
            "fk": {"max_fk": score["max_fk"], "pct_in_band": score["pct_in_band"],
                   "whole_passage_fk": score["whole_passage_fk"], "readability_pass_v3": True},
            "accuracy": {"score": acc_score, "note": note},
            "rewrite_iters": iters, "n_sentences": n_sent,
        }
        kept_records.append(rec)
        stats["kept"] += 1
        print(f"[{i}/{len(items)}] KEEP (acc=2, {iters} rw, max_fk={score['max_fk']}, "
              f"band={score['pct_in_band']}) | {phrasing}")

    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- yield report ---
    n = stats["n_items"]
    avg_iters = stats["rewrite_iters_total"] / n if n else 0
    print("\n=== YIELD REPORT ===")
    print(f"items attempted:            {n}")
    print(f"discarded (readability):    {stats['discarded_readability']}")
    print(f"passed readability:         {stats['readability_passers']}")
    print(f"discarded (format 4-8):     {stats['discarded_format']}")
    print(f"discarded (accuracy != 2):  {stats['discarded_accuracy']}")
    print(f"accuracy histogram:         {stats['accuracy_hist']}")
    if n:
        print(f"KEPT (final):               {stats['kept']}  ({stats['kept']/n*100:.0f}% yield)")
    print(f"avg rewrite iterations:     {avg_iters:.2f}")
    print(f"\nWrote {out_path} ({stats['kept']} examples)")

    stats_path = out_path.with_suffix(".stats.json")
    stats["avg_rewrite_iters"] = round(avg_iters, 2)
    stats["yield_rate"] = round(stats["kept"] / n, 3) if n else 0
    stats["mode"] = mode
    stats["teacher"] = args.teacher
    stats["judge"] = args.judge
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
