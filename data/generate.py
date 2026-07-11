"""Part A.3-A.5 - the generate -> readability-rewrite -> accuracy-gate pipeline.

For each TRAIN (concept, phrasing) pair:
  1. Teacher writes a candidate explanation, few-shot-anchored on data/exemplars.json.
  2. Score every sentence with the litmus FK harness (score_text).
  3. If it fails the readability spec (v4 gate: whole-passage FK 3-6 AND ARI 3-7,
     even dispersion, no long run-on over the backstop), send DIRECTION-AWARE
     feedback back to the teacher - a passage that reads too HARD overall gets
     "make simpler", one that reads too EASY/choppy gets "combine/enrich", and the
     hardest/easiest outlier sentences are named. Repeat up to --max-rewrites.
     [Day-2 bug fixed: the old feedback always said "make simpler", which pushed
     already-too-easy text further below the band and yielded 0% survival.]
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from litmus.accuracy import build_judge_prompt
from litmus.env import make_client
from litmus.fk_score import (
    LONG_MIN_WORDS,
    score_text,
)
from data.v4r3 import (
    TARGETED_V4R3_ITEMS,
    V4R3_ARI_BAND,
    V4R3_DISP_MAX,
    V4R3_FK_BAND,
    V4R3_MAX_SENTENCE_FK,
    V4R3_MAX_SENTENCES,
    V4R3_MIN_SENTENCES,
    accuracy_is_2,
    meets_target,
    norm_text,
    target_config,
    training_prompt,
    with_current_fk,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
CONCEPTS_PATH = HERE / "concepts.json"
EXEMPLARS_PATH = HERE / "exemplars.json"

MIN_SENTENCES = V4R3_MIN_SENTENCES
MAX_SENTENCES = V4R3_MAX_SENTENCES

# --- v4r3 generation target: TIGHTER than the eval gate ------------------------
# The eval gate (litmus.fk_score.readability_pass_v4) is FK 3.0-6.0 / ARI 3.0-7.0 /
# dispersion<=1.7 and stays UNCHANGED. v4r3 trains on data centered tighter INSIDE
# the eval band so the model's output spread can still land inside the wider gate.
# Anything passing this target also passes the eval gate (strict subset).
GEN_FK_BAND = V4R3_FK_BAND
GEN_ARI_BAND = V4R3_ARI_BAND
GEN_DISP_MAX = V4R3_DISP_MAX
GEN_MAX_SENTENCE_FK = V4R3_MAX_SENTENCE_FK


def meets_gen_target(score: dict, target: dict | None = None) -> bool:
    """v4r3 DATA-GENERATION acceptance: the tighter, centered band above.

    Used ONLY to decide which generated examples to keep. Evaluation still uses the
    unchanged litmus readability_pass_v4.
    """
    return meets_target(score, target)


GEN_SYSTEM = (
    "You write science explanations for a curious 8-to-9-year-old in 3rd grade.\n"
    "Follow ALL of these rules:\n"
    "- Write at a REAL 3rd-grade reading level (not baby-talk). Use words a 3rd "
    "grader knows; you may reach for a longer, more precise word when it is the "
    "right one, as long as the sentence around it makes its meaning clear.\n"
    "- Aim for sentences of about 10 to 16 words. Do NOT write choppy 3-6 word "
    "sentences (they read as baby-talk) and do NOT write long, winding run-ons.\n"
    "- Avoid technical jargon; if a real term is truly needed, name it and explain "
    "it in plain words in the same breath.\n"
    "- Be scientifically ACCURATE and explain the real HOW/WHY (the mechanism), not "
    "just a definition. Never oversimplify into something that becomes wrong.\n"
    "- When you make wording simpler, NEVER drop the cause-and-effect. Keep the real "
    "HOW/WHY intact - a simpler sentence that loses the mechanism is WRONG, not simpler.\n"
    "- Aim for the SOLID MIDDLE of 3rd grade, evenly across every sentence — neither "
    "so plain it becomes baby-talk nor so dense it drifts toward 4th grade.\n"
    "- Write 4 to 6 sentences. Return ONLY the explanation text, no preamble, no title."
)


def few_shot_block(exemplars: list[dict]) -> str:
    lines = ["Here are examples of the target style:\n"]
    for e in exemplars:
        lines.append(f"Concept: {e['concept']}\nExplanation: {e['explanation']}\n")
    return "\n".join(lines)


def fk_feedback(text: str, score: dict, target: dict | None = None) -> str:
    """DIRECTION-AWARE rewrite instruction, aimed at the v4r3 generation target.

    Targets the tighter GEN band, NOT the
    wider eval gate, so the rewrite loop pushes candidates toward band-CENTER. The
    *direction* is set by which edge the passage as a whole violates: too HARD ->
    simplify; too EASY/choppy -> enrich. Per-sentence outliers (hardest / easiest
    sentences, and any long run-on the backstop caught) are named so the teacher
    knows where to act. Fixing only one direction is what tanked Day-2 yield.
    """
    target = target or target_config()
    wp_fk = score["whole_passage_fk"]
    wp_ari = score["whole_passage_ari"]
    fk_lo, fk_hi = target["fk_band"]
    ari_lo, ari_hi = target["ari_band"]

    too_hard = [r for r in score["sentences"] if r["fk"] > fk_hi]
    too_easy = [r for r in score["sentences"] if r["fk"] < fk_lo]
    long_over = [r for r in score["sentences"]
                 if r["words"] >= LONG_MIN_WORDS and r["fk"] > target["max_sentence_fk"]]

    over = wp_fk > fk_hi or wp_ari > ari_hi
    under = wp_fk < fk_lo or wp_ari < ari_lo
    uneven = score["fk_stdev"] > target["disp_max"]

    bits = [
        f"The reading level is off. Target: whole-passage Flesch-Kincaid grade "
        f"{fk_lo}-{fk_hi} AND ARI {ari_lo}-{ari_hi}, written EVENLY (no sentence much "
        f"harder than the rest). Right now this passage is FK {wp_fk}, ARI {wp_ari}."
    ]
    if score["max_fk"] > target["max_sentence_fk"]:
        bits.append(
            f"\nAt least one sentence is too hard. Keep every sentence at FK "
            f"{target['max_sentence_fk']} or below."
        )
    if over:
        bits.append("\nIt reads TOO HARD overall. Bring it down with shorter, more "
                     "common words and by splitting the longest sentences - WITHOUT "
                     "dropping any science or flattening it into baby-talk.")
    if under:
        bits.append("\nIt reads TOO EASY / choppy overall - below a real 3rd-grade "
                     "level. Combine short sentences with a neighbor and add a little "
                     "concrete detail so it reads like natural grade-3 informational "
                     "writing (sentences ~10-16 words), WITHOUT adding hard, technical "
                     "words.")
    if uneven and not (over or under):
        bits.append("\nThe level is UNEVEN - some sentences are much harder than "
                     "others. Even them out so every sentence sits at about the same "
                     "grade-3 reading level.")
    if too_hard:
        bits.append("\nHardest sentences (simplify or split these):")
        for r in too_hard:
            bits.append(f'  - FK {r["fk"]}: "{r["sentence"]}"')
    if long_over:
        bits.append("\nThese long sentences are run-ons over the limit - split each "
                     "into two shorter ones:")
        for r in long_over:
            bits.append(f'  - FK {r["fk"]} ({r["words"]} words): "{r["sentence"]}"')
    if too_easy and (under or uneven):
        bits.append("\nThese sentences are too simple / choppy - combine or enrich "
                     "them (keep them grade-3, ~10-16 words):")
        for r in too_easy:
            flag = " (very short)" if r["short_flag"] else ""
            bits.append(f'  - FK {r["fk"]}{flag}: "{r["sentence"]}"')
    bits.append(
        "\nRewrite the WHOLE explanation, keeping every science fact correct and the "
        "mechanism intact. Stay at 4-6 sentences. Return only the new explanation."
    )
    return "\n".join(bits)


def _content(resp) -> str:
    """Pull non-empty message text out of a completion, or RAISE.

    Some gateway-routed models (Gemini has done this on this project's gateway)
    intermittently return null/empty content. Raising here — instead of crashing
    on ``None.strip()`` — lets ``call_with_retry`` treat it as a transient blip
    and retry, which is exactly what we want under concurrency.
    """
    choices = getattr(resp, "choices", None)
    if not choices:
        raise RuntimeError("model returned no choices")
    content = choices[0].message.content
    if content is None or not content.strip():
        raise RuntimeError("model returned empty content")
    return content.strip()


def call_with_retry(fn, *, what, tag, log, tries=5, base_delay=2.0, max_delay=30.0):
    """Run ``fn`` with exponential backoff. Retries any exception (rate limits,
    transient 5xx, empty-content) up to ``tries`` times, then re-raises so the
    caller can drop just that one item instead of killing the whole run."""
    delay = base_delay
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - deliberately broad: any API hiccup retries
            if attempt >= tries:
                log(f"{tag} {what}: FAILED after {tries} tries ({type(e).__name__}: {e})")
                raise
            log(f"{tag} {what}: {type(e).__name__}: {e} — retry {attempt}/{tries} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


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
    return _content(resp)


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
    return _content(resp)


def judge_accuracy(client, judge_model, concept, text):
    """0/1/2 mechanism-rubric judge. Judge != student; different model from teacher."""
    resp = client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user",
                   "content": build_judge_prompt(concept, text, audience_calibrated=True)}],
    )
    data = json.loads(_content(resp))
    return int(data["score"]), data.get("justification", "")


def accuracy_feedback(concept: str, text: str, judge_note: str) -> str:
    """Mechanism-REPAIR instruction for a draft that already reads at grade level but
    lost/weakened the science (accuracy judge scored < 2).

    The whole point of A3: the draft is readable, so do NOT re-simplify it. Restore the
    real cause-and-effect the judge flagged, at the SAME grade-3 reading level, without
    adding hard words. Fixing accuracy by making it harder to read would just bounce it
    back out of the readability band.
    """
    note = (judge_note or "").strip()
    bits = [
        f'The explanation for "{concept}" reads at the right grade-3 level, but it is '
        "not accurate enough: it is missing or has weakened the real cause-and-effect "
        "(the actual HOW/WHY of the mechanism)."
    ]
    if note:
        bits.append(f"\nThe accuracy check said: {note}")
    bits.append(
        "\nRewrite it so it explains the true mechanism (the real why it happens), while "
        "keeping the SAME easy grade-3 reading level - short, common words, sentences of "
        "about 10-16 words. Do NOT add hard or technical words, and do NOT make it read "
        "harder. Stay at 4-6 sentences. Return only the new explanation."
    )
    return "\n".join(bits)


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


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def pick_teacher(models: list[str], item_index: int) -> str:
    return models[(item_index - 1) % len(models)]


def pick_rewriter(rewriters: list[str], teachers: list[str], teacher: str, item_index: int,
                  rewrite_iter: int) -> str:
    if rewriters:
        return rewriters[(item_index + rewrite_iter - 2) % len(rewriters)]
    if len(teachers) > 1:
        idx = teachers.index(teacher)
        return teachers[(idx + 1) % len(teachers)]
    return teacher


def eval_prompt_keys(cdata: dict) -> set[str]:
    return {norm_text(item) for item in cdata.get("eval", [])}


def dedupe_work_items(items: list[dict], seen_prompts: set[str], eval_prompts: set[str]) -> list[dict]:
    out = []
    for item in items:
        key = norm_text(item["phrasing"])
        if key in seen_prompts or key in eval_prompts:
            continue
        seen_prompts.add(key)
        out.append(item)
    return out


def read_jsonl(path: Path) -> list[dict]:
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_seed_records(seed_paths: list[str], target: dict, eval_prompts: set[str],
                      min_sentences: int, max_sentences: int) -> tuple[list[dict], dict, set[str]]:
    stats = {
        "seed_seen": 0,
        "seed_kept": 0,
        "seed_discarded_eval": 0,
        "seed_discarded_duplicate": 0,
        "seed_discarded_readability": 0,
        "seed_discarded_format": 0,
        "seed_discarded_accuracy": 0,
    }
    seen_prompts: set[str] = set()
    kept = []
    for raw_path in seed_paths:
        for rec in read_jsonl(Path(raw_path)):
            stats["seed_seen"] += 1
            prompt = training_prompt(rec)
            key = norm_text(prompt)
            if key in eval_prompts or norm_text(rec["concept"]) in eval_prompts:
                stats["seed_discarded_eval"] += 1
                continue
            if key in seen_prompts:
                stats["seed_discarded_duplicate"] += 1
                continue
            if not accuracy_is_2(rec):
                stats["seed_discarded_accuracy"] += 1
                continue
            score = score_text(rec["explanation"])
            if not meets_gen_target(score, target):
                stats["seed_discarded_readability"] += 1
                continue
            if score["n_sentences"] < min_sentences or score["n_sentences"] > max_sentences:
                stats["seed_discarded_format"] += 1
                continue
            seen_prompts.add(key)
            seed_rec = with_current_fk(rec, score, target)
            seed_rec.setdefault("teacher", "seed-existing")
            seed_rec.setdefault("rewriters", [])
            seed_rec.setdefault("judge", "seed-existing")
            seed_rec["seed_source"] = str(raw_path)
            kept.append(seed_rec)
            stats["seed_kept"] += 1
    return kept, stats, seen_prompts


def run_authored(client, authored_path, judge_model, out_path, target, min_sentences, max_sentences):
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

        if not meets_gen_target(score, target):
            stats["discarded_readability"] += 1
            print(f"[{i}/{len(recs)}] DISCARD readability (wp_fk={score.get('whole_passage_fk')}, "
                  f"ari={score.get('whole_passage_ari')}, stdev={score.get('fk_stdev')}, "
                  f"long_over={score.get('n_long_over_ceiling')}) | {phrasing}")
            continue
        stats["readability_passers"] += 1

        n_sent = score["n_sentences"]
        if n_sent < min_sentences or n_sent > max_sentences:
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
            "fk": {"max_fk": score["max_fk"], "whole_passage_fk": score["whole_passage_fk"],
                   "whole_passage_ari": score["whole_passage_ari"], "fk_stdev": score["fk_stdev"],
                   "readability_pass_v4": True},
            "accuracy": {"score": acc_score, "note": note},
            "rewrite_iters": rec.get("rewrite_iters", 0), "n_sentences": n_sent,
            "teacher": "claude", "judge": judge_model, "generation_target": target,
        })
        stats["kept"] += 1
        print(f"[{i}/{len(recs)}] KEEP (acc=2, wp_fk={score['whole_passage_fk']}, "
              f"ari={score['whole_passage_ari']}, {n_sent} sent) | {phrasing}")

    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = stats["n_items"]
    print("\n=== YIELD REPORT (authored) ===")
    print(f"items:                     {n}")
    print(f"discarded (readability):   {stats['discarded_readability']}")
    print(f"discarded (format):        {stats['discarded_format']}")
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
    stats["generation_target"] = target
    stats["min_sentences"] = min_sentences
    stats["max_sentences"] = max_sentences
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {stats_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="gpt-4o", help="generation/rewrite model (frontier)")
    ap.add_argument("--teachers", default=None,
                    help="comma-separated generation models; overrides --teacher")
    ap.add_argument("--rewriters", default=None,
                    help="comma-separated rewrite models; default uses opposite --teachers entry")
    ap.add_argument("--judge", default="gpt-4o-mini", help="accuracy judge (!= student, != teacher)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-rewrites", type=int, default=4, help="readability rewrite cap")
    ap.add_argument("--max-accuracy-repairs", type=int, default=2,
                    help="mechanism-repair rewrites for a readable-but-inaccurate draft, "
                    "on TOP of --max-rewrites (A3 joint gate). 0 = old behavior (terminal "
                    "accuracy gate, no repair).")
    ap.add_argument("--limit", type=int, default=None, help="cap number of work items")
    ap.add_argument("--target-kept", type=int, default=None,
                    help="stop once this many total records are kept, including seeds")
    ap.add_argument("--seed", default=None,
                    help="comma-separated JSONL seed files to filter/dedupe into the output first")
    ap.add_argument("--fk-min", type=float, default=GEN_FK_BAND[0])
    ap.add_argument("--fk-max", type=float, default=GEN_FK_BAND[1])
    ap.add_argument("--ari-min", type=float, default=GEN_ARI_BAND[0])
    ap.add_argument("--ari-max", type=float, default=GEN_ARI_BAND[1])
    ap.add_argument("--disp-max", type=float, default=GEN_DISP_MAX)
    ap.add_argument("--max-sentence-fk", type=float, default=GEN_MAX_SENTENCE_FK)
    ap.add_argument("--min-sentences", type=int, default=MIN_SENTENCES)
    ap.add_argument("--max-sentences", type=int, default=MAX_SENTENCES)
    ap.add_argument("--no-targeted-v4r3", action="store_true",
                    help="do not prepend targeted near-neighbor work items")
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
    ap.add_argument("--concurrency", type=int, default=8,
                    help="items processed in parallel. The loop is I/O-bound on API "
                    "calls, so this is a near-linear speedup until the gateway rate-limits.")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out and SKIP prompts already present. Combined with "
                    "the per-record incremental write, a killed run restarts where it left off.")
    args = ap.parse_args()
    target = target_config(
        fk_min=args.fk_min,
        fk_max=args.fk_max,
        ari_min=args.ari_min,
        ari_max=args.ari_max,
        disp_max=args.disp_max,
        max_sentence_fk=args.max_sentence_fk,
    )

    if args.authored:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Claude is the teacher here, so default the judge to the strong, different
        # family (gpt-4o) unless the caller overrode it.
        judge = args.judge if args.judge != ap.get_default("judge") else "gpt-4o"
        run_authored(make_client(), args.authored, judge, out_path, target,
                     args.min_sentences, args.max_sentences)
        return

    cdata = json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))
    eval_prompts = eval_prompt_keys(cdata)
    seed_paths = parse_csv(args.seed)
    kept_records, seed_stats, seen_prompts = load_seed_records(
        seed_paths, target, eval_prompts, args.min_sentences, args.max_sentences,
    )
    items = build_work_items(cdata, args.junk)
    if not args.no_targeted_v4r3:
        items = TARGETED_V4R3_ITEMS + items
    items = dedupe_work_items(items, seen_prompts, eval_prompts)
    if args.sample:
        items = sample_items(items, args.sample)
    if args.limit:
        items = items[: args.limit]
    exemplars = json.loads(EXEMPLARS_PATH.read_text(encoding="utf-8"))["exemplars"]
    fewshot = few_shot_block(exemplars)

    client = make_client()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "n_items": len(items),
        "kept": len(kept_records),
        "generated_attempted": 0,
        "generated_kept": 0,
        "discarded_readability": 0,
        "discarded_format": 0,
        "discarded_accuracy": 0,
        "errors": 0,
        "accuracy_hist": {0: 0, 1: 0, 2: 0},
        "rewrite_iters_total": 0,
        "accuracy_repairs_total": 0,
        "readability_passers": 0,
        **seed_stats,
    }

    mode = "JUNK (no filtering)" if args.junk else ("REVIEW SAMPLE" if args.sample else "FULL pipeline")
    teachers = parse_csv(args.teachers) or [args.teacher]
    rewriters = parse_csv(args.rewriters)

    # --- concurrency + thread-safe logging/writing/stats -----------------------
    print_lock = threading.Lock()
    write_lock = threading.Lock()
    stats_lock = threading.Lock()
    stop_event = threading.Event()   # set once target_kept is reached; drains in-flight
    t0 = time.monotonic()

    def log(msg: str) -> None:
        with print_lock:
            print(msg, flush=True)

    # --- resume + incremental (crash-safe) output ------------------------------
    # Records are written the instant they pass, not at the end. With --resume we
    # read what's already there, skip those prompts, and append. A kill now costs
    # at most the handful of items in flight — never the whole run.
    resume = bool(args.resume and out_path.exists())
    done_keys: set[str] = set()
    if resume:
        for rec in read_jsonl(out_path):
            done_keys.add(norm_text(training_prompt(rec)))
        log(f"[resume] {len(done_keys)} records already in {out_path.name}; skipping those prompts")
    out_f = out_path.open("a" if resume else "w", encoding="utf-8")

    def append_records(records: list[dict]) -> None:
        if not records:
            return
        with write_lock:
            for r in records:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()

    # seeds go to disk first, so they survive a crash mid-generation
    new_seeds = [s for s in kept_records if norm_text(training_prompt(s)) not in done_keys]
    append_records(new_seeds)
    for s in new_seeds:
        done_keys.add(norm_text(training_prompt(s)))
    stats["kept"] = len(done_keys) if resume else len(new_seeds)

    # never re-generate a prompt already on disk (seed or prior partial run)
    items = [it for it in items if norm_text(it["phrasing"]) not in done_keys]
    stats["n_items"] = len(items)

    print(f"=== data-gen: {mode} (concurrency={args.concurrency}) ===", flush=True)
    print(f"teachers={teachers}  rewriters={rewriters or 'auto'}  judge={args.judge}  "
          f"items={len(items)}  seed_kept={seed_stats['seed_kept']}  "
          f"kept_so_far={stats['kept']}  target_kept={args.target_kept}  "
          f"max_rewrites={args.max_rewrites}  resume={resume}", flush=True)
    print(f"generation_target={target}\n", flush=True)

    def progress_suffix() -> str:
        elapsed = time.monotonic() - t0
        attempts = stats["generated_attempted"]
        rate = attempts / elapsed if elapsed > 0 else 0.0
        s = f"kept={stats['kept']}"
        if args.target_kept:
            s += f"/{args.target_kept}"
        s += f" attempts={attempts} {rate * 60:.1f}/min {elapsed / 60:.1f}min"
        if args.target_kept and attempts > 0:
            yr = stats["generated_kept"] / attempts
            remaining = max(0, args.target_kept - stats["kept"])
            if yr > 0 and rate > 0:
                s += f" ETA~{(remaining / yr / rate) / 60:.0f}min"
        return s

    def process_item(i: int, item: dict) -> dict:
        if stop_event.is_set():
            return {"status": "skipped"}
        concept, phrasing = item["concept"], item["phrasing"]
        tag = f"[{i}]"
        teacher_model = pick_teacher(teachers, i)
        try:
            text = call_with_retry(
                lambda: generate_candidate(client, teacher_model, phrasing, fewshot, args.temperature),
                what="generate", tag=tag, log=log)
            score = score_text(text)

            if args.junk:
                rec = {
                    "concept": concept, "phrasing": phrasing, "explanation": text,
                    "fk": {"max_fk": score.get("max_fk"), "whole_passage_fk": score.get("whole_passage_fk"),
                           "whole_passage_ari": score.get("whole_passage_ari"),
                           "fk_stdev": score.get("fk_stdev"),
                           "readability_pass_v4": score.get("readability_pass_v4")},
                    "accuracy": {"score": None, "note": "junk-mode: accuracy gate skipped"},
                    "rewrite_iters": 0, "n_sentences": score.get("n_sentences"),
                    "teacher": teacher_model, "judge": None, "generation_target": target,
                }
                return {"status": "keep", "rec": rec, "phrasing": phrasing, "iters": 0, "score": score}

            # --- joint readability + accuracy loop (A3) --------------------------
            # Rewrite until BOTH gates pass or both budgets are spent. Readability
            # misses get direction-aware fk_feedback (readability rewrites, capped at
            # --max-rewrites); a readable-but-inaccurate draft gets accuracy_feedback
            # that restores the mechanism WITHOUT raising the reading level (repair
            # rewrites, capped separately at --max-accuracy-repairs). Accuracy is judged
            # only on readable drafts, and the score is cached (acc_score=None means the
            # current text is unjudged) so unchanged text is never re-judged.
            read_iters = 0
            repairs = 0
            rewriter_history: list[str] = []
            acc_score, note = None, ""

            while True:
                if meets_gen_target(score, target):
                    if acc_score is None:
                        acc_score, note = call_with_retry(
                            lambda: judge_accuracy(client, args.judge, phrasing, text),
                            what="judge", tag=tag, log=log)
                    if acc_score == 2:
                        break                                   # both gates pass
                    if repairs >= args.max_accuracy_repairs:
                        break                                   # readable but can't fix accuracy
                    feedback = accuracy_feedback(phrasing, text, note)
                    repairs += 1
                    kind = "acc-repair"
                else:
                    if read_iters >= args.max_rewrites:
                        break                                   # can't reach the band
                    feedback = fk_feedback(text, score, target)
                    read_iters += 1
                    kind = "readability"
                step = read_iters + repairs
                rewriter_model = pick_rewriter(rewriters, teachers, teacher_model, i, step)
                rewriter_history.append(rewriter_model)
                text = call_with_retry(
                    lambda rm=rewriter_model, prev=text, fb=feedback: rewrite_candidate(
                        client, rm, phrasing, prev, fb, fewshot, args.temperature),
                    what=f"rewrite#{step}:{kind}", tag=tag, log=log)
                score = score_text(text)
                acc_score = None                                # text changed -> accuracy stale

            iters = read_iters + repairs

            if not meets_gen_target(score, target):
                return {"status": "discard_readability", "phrasing": phrasing,
                        "iters": iters, "repairs": repairs, "score": score}

            n_sent = score["n_sentences"]
            if n_sent < args.min_sentences or n_sent > args.max_sentences:
                return {"status": "discard_format", "phrasing": phrasing,
                        "iters": iters, "repairs": repairs, "n_sent": n_sent}

            if acc_score is None:                               # last edit was a readability fix
                acc_score, note = call_with_retry(
                    lambda: judge_accuracy(client, args.judge, phrasing, text),
                    what="judge", tag=tag, log=log)
            if acc_score != 2:
                return {"status": "discard_accuracy", "phrasing": phrasing,
                        "iters": iters, "repairs": repairs, "acc": acc_score}

            rec = {
                "concept": concept, "phrasing": phrasing, "explanation": text,
                "fk": {"max_fk": score["max_fk"], "whole_passage_fk": score["whole_passage_fk"],
                       "whole_passage_ari": score["whole_passage_ari"], "fk_stdev": score["fk_stdev"],
                       "readability_pass_v4": True},
                "accuracy": {"score": acc_score, "note": note},
                "rewrite_iters": iters, "accuracy_repairs": repairs, "n_sentences": n_sent,
                "teacher": teacher_model, "rewriters": rewriter_history, "judge": args.judge,
                "generation_target": target,
            }
            return {"status": "keep", "rec": rec, "phrasing": phrasing,
                    "iters": iters, "repairs": repairs, "score": score}
        except Exception as e:  # noqa: BLE001 - a single failed item must not kill the run
            return {"status": "error", "phrasing": phrasing, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(process_item, i, item) for i, item in enumerate(items, 1)]
        for fut in as_completed(futures):
            res = fut.result()
            st = res["status"]
            if st == "skipped":
                continue
            phrasing = res.get("phrasing", "")
            with stats_lock:
                stats["generated_attempted"] += 1
                stats["rewrite_iters_total"] += res.get("iters", 0)
                stats["accuracy_repairs_total"] += res.get("repairs", 0)
                if st == "keep":
                    append_records([res["rec"]])
                    stats["kept"] += 1
                    stats["generated_kept"] += 1
                    stats["readability_passers"] += 1
                    if not args.junk:
                        stats["accuracy_hist"][2] = stats["accuracy_hist"].get(2, 0) + 1
                    sc = res["score"]
                    log(f"[KEEP] rw={res['iters']} rep={res.get('repairs', 0)} "
                        f"wp_fk={sc.get('whole_passage_fk')} ari={sc.get('whole_passage_ari')} "
                        f"| {phrasing}  ::  {progress_suffix()}")
                    if args.target_kept is not None and stats["kept"] >= args.target_kept and not stop_event.is_set():
                        stop_event.set()
                        log(f"*** target kept reached ({stats['kept']}/{args.target_kept}); "
                            f"draining {args.concurrency - 1} in-flight items then stopping ***")
                elif st == "discard_readability":
                    stats["discarded_readability"] += 1
                    sc = res["score"]
                    log(f"[DISCARD readability] wp_fk={sc.get('whole_passage_fk')} "
                        f"ari={sc.get('whole_passage_ari')} stdev={sc.get('fk_stdev')} "
                        f"rw={res['iters']} | {phrasing}")
                elif st == "discard_format":
                    stats["readability_passers"] += 1
                    stats["discarded_format"] += 1
                    log(f"[DISCARD format] {res['n_sent']} sentences | {phrasing}")
                elif st == "discard_accuracy":
                    stats["readability_passers"] += 1
                    acc = res["acc"]
                    stats["accuracy_hist"][acc] = stats["accuracy_hist"].get(acc, 0) + 1
                    stats["discarded_accuracy"] += 1
                    log(f"[DISCARD accuracy={acc}] rw={res['iters']} | {phrasing}")
                elif st == "error":
                    stats["errors"] += 1
                    log(f"[ERROR] {res.get('error')} | {phrasing}")

    out_f.close()

    # --- yield report ---
    n = stats["generated_attempted"]
    avg_iters = stats["rewrite_iters_total"] / n if n else 0
    print("\n=== YIELD REPORT ===")
    print(f"seed kept:                  {stats['seed_kept']}")
    print(f"generated items attempted:  {n}")
    print(f"discarded (readability):    {stats['discarded_readability']}")
    print(f"passed readability:         {stats['readability_passers']}")
    print(f"discarded (format):         {stats['discarded_format']}")
    print(f"discarded (accuracy != 2):  {stats['discarded_accuracy']}")
    print(f"errors (dropped):           {stats['errors']}")
    print(f"accuracy histogram:         {stats['accuracy_hist']}")
    print(f"accuracy repairs (A3):      {stats['accuracy_repairs_total']}")
    if n:
        print(f"generated kept:             {stats['generated_kept']}  "
              f"({stats['generated_kept']/n*100:.0f}% generated yield)")
    print(f"KEPT (final):               {stats['kept']}")
    print(f"avg rewrite iterations:     {avg_iters:.2f}")
    print(f"wall time:                  {(time.monotonic() - t0) / 60:.1f} min")
    print(f"\nWrote {out_path} ({stats['kept']} examples)", flush=True)

    stats_path = out_path.with_suffix(".stats.json")
    stats["avg_rewrite_iters"] = round(avg_iters, 2)
    stats["avg_accuracy_repairs"] = round(stats["accuracy_repairs_total"] / n, 2) if n else 0
    stats["generated_yield_rate"] = round(stats["generated_kept"] / n, 3) if n else 0
    stats["max_accuracy_repairs"] = args.max_accuracy_repairs
    stats["mode"] = mode
    stats["teachers"] = teachers
    stats["rewriters"] = rewriters or "auto-opposite-teacher"
    stats["judge"] = args.judge
    stats["generation_target"] = target
    stats["target_kept"] = args.target_kept
    stats["min_sentences"] = args.min_sentences
    stats["max_sentences"] = args.max_sentences
    stats["concurrency"] = args.concurrency
    stats["wall_time_min"] = round((time.monotonic() - t0) / 60, 1)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()
