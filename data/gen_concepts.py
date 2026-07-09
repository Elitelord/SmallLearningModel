"""Part A.1 / Day-3 Step 1 - scale the concept list, split, and phrase.

Produces data/concepts.json:
    {
      "meta": {...},
      "eval":        [24 held-out concepts = 12 litmus + 12 extra],
      "eval_litmus": [the 12 litmus concepts (subset of eval)],
      "train_concepts": [~250-350 trainable concepts],
      "phrasings":   { "<train concept>": ["<phrasing1>", "<phrasing2>", ...] }  # 2-4 each
    }

Guardrails:
  - NARROW DOMAIN: elementary physical + life science only.
  - NO LEAKAGE: eval concepts (litmus 12 + 12 extra) AND anything that normalizes
    to them are removed from train and never phrased for training.
  - Phrasings teach the BEHAVIOR (robustness to wording), not concept->text memorization.

Usage:
    .venv\\Scripts\\python -m data.gen_concepts --target 300           # real run
    .venv\\Scripts\\python -m data.gen_concepts --target 60 --offline  # seed list, no API
    .venv\\Scripts\\python -m data.gen_concepts --target 300 --no-phrasings
"""

import argparse
import json
import re
import sys
from pathlib import Path

from litmus.concepts import CONCEPTS as LITMUS_CONCEPTS
from litmus.env import load_env

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_PATH = Path(__file__).resolve().parent / "concepts.json"

DOMAIN = (
    "elementary physical science and life science that a curious 6-to-9-year-old "
    "would ask about (light, heat, water, weather, simple forces, motion, the sky, "
    "plants, animals, the human body, materials, sound)"
)

# 12 EXTRA held-out eval concepts (with the litmus 12 -> 24 eval concepts).
# Chosen distinct from the litmus set AND from the few-shot exemplar concepts
# (sweat / eat food / shadow / seed), so nothing in eval leaks via the anchors.
EXTRA_EVAL_CONCEPTS = [
    "Why do leaves change color in the fall?",
    "How do bees help plants grow?",
    "Why does the ocean taste salty?",
    "Why do we get goosebumps?",
    "How does a caterpillar turn into a butterfly?",
    "Why does bread rise when we bake it?",
    "Why do stars twinkle at night?",
    "How do our ears help us hear?",
    "What makes thunder so loud?",
    "Why do some things float while others sink?",
    "How do birds fly?",
    "Why do we shiver when we are cold?",
]

# Exemplar concepts (few-shot anchors) — keep out of BOTH train and eval to avoid
# the model just parroting the anchors and to keep eval clean.
EXEMPLAR_CONCEPTS = [
    "Why do we sweat when it is hot?",
    "Why do we need to eat food?",
    "What is a shadow?",
    "How does a seed grow into a plant?",
]

SEED_CONCEPTS = [
    "Why does the moon come out at night?", "How do our hearts pump blood?",
    "Why do we have fingernails?", "How does a plant drink water?",
    "Why do we get hungry?", "What makes popcorn pop?", "Why do dogs pant?",
    "How does a boat stay on top of the water?", "Why do we have bones?",
    "How do spiders make their webs?", "Why do apples turn brown after we cut them?",
    "Why do we blink our eyes?", "How does a magnet pick up metal?",
    "Why does milk go bad if we leave it out?", "How does a frog catch a fly?",
    "Why does snow melt in our hands?", "Why is grass green?",
    "How do fish stay warm in cold water?", "Why do we have to brush our teeth?",
    "What makes a rainbow of colors in soap bubbles?", "Why do birds build nests?",
    "How does ice cream melt so fast?", "Why do we cough when we are sick?",
    "How do worms help the soil?", "Why does a shadow get longer at sunset?",
    "Why do cats purr?", "How does a flower turn into a fruit?",
    "Why do we feel dizzy when we spin?", "What makes the leaves fall off trees?",
    "How do seeds travel to new places?", "Why does the sea have tides?",
    "Why do we have two eyes?", "How does a candle make light?",
    "Why do puddles freeze in winter?", "How do ants find their food?",
    "Why does hot air rise?", "Why do we get tired at night?",
    "How does a bird know when to fly south?", "Why do bruises turn colors?",
    "What makes a ball bounce?",
]


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _dedup_exclude(candidates, exclude_norms):
    seen = set(exclude_norms)
    out = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        n = _norm(c)
        if n in seen:
            continue
        seen.add(n)
        out.append(c)
    return out


def generate_concepts(client, model, target, exclude, batch=60, temperature=1.0):
    exclude_norms = {_norm(c) for c in exclude}
    collected, seen = [], set(exclude_norms)
    rounds = 0
    while len(collected) < target and rounds < 14:
        rounds += 1
        avoid = list(exclude) + collected[-40:]
        prompt = (
            f"List {batch} distinct science questions about {DOMAIN}.\n"
            "Each must be a short, natural question a young child would ask, ending in '?'.\n"
            "Vary the topics widely; do not cluster on one theme.\n"
            "Do NOT repeat or paraphrase any of these already-used questions:\n"
            + "\n".join(f"- {a}" for a in avoid)
            + '\n\nReturn ONLY a JSON object: {"concepts": ["...", "..."]}'
        )
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        fresh = _dedup_exclude(json.loads(resp.choices[0].message.content).get("concepts", []), seen)
        for c in fresh:
            seen.add(_norm(c))
        collected.extend(fresh)
        print(f"  concepts round {rounds}: +{len(fresh)} -> {len(collected)}/{target}")
        if not fresh:
            break
    return collected[:target]


def generate_phrasings(client, model, concepts, batch=20, per=3, temperature=0.9):
    """2-4 phrasings per concept (incl. the original). Batched for cost."""
    phrasings = {}
    for start in range(0, len(concepts), batch):
        chunk = concepts[start:start + batch]
        prompt = (
            f"For each science question below, give {per} DIFFERENT natural rephrasings "
            "a child (or parent) might use to ask the same thing. Vary the wording and "
            "sentence shape (e.g. 'Why is X?', 'How come X?', 'Can you explain X', "
            "'What makes X happen?'). Keep the meaning identical.\n\n"
            + "\n".join(f"{i+1}. {c}" for i, c in enumerate(chunk))
            + '\n\nReturn ONLY JSON: {"items": [{"original": "...", "phrasings": ["...", ...]}]}'
        )
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.choices[0].message.content)
        returned = {_norm(it.get("original", "")): it.get("phrasings", [])
                    for it in data.get("items", [])}
        for c in chunk:
            extra = returned.get(_norm(c), [])
            # original first, then dedup extras that don't collapse to the original
            variants = [c]
            seen = {_norm(c)}
            for p in extra:
                if _norm(p) and _norm(p) not in seen:
                    variants.append(p.strip())
                    seen.add(_norm(p))
            phrasings[c] = variants[:4]  # cap 2-4
        print(f"  phrasings {start+len(chunk)}/{len(concepts)}")
    return phrasings


def scrub_leakage(phrasings: dict, forbidden: list) -> tuple[dict, int, int]:
    """Remove any phrasing that normalizes to a forbidden (eval/exemplar) concept,
    and drop any train concept whose base itself collides. Phrasings can rephrase
    a train concept straight onto a held-out eval concept — this catches that."""
    forbid = {_norm(c) for c in forbidden}
    clean, dropped_ph, dropped_concepts = {}, 0, 0
    for c, plist in phrasings.items():
        if _norm(c) in forbid:
            dropped_concepts += 1
            continue
        kept = []
        for p in plist:
            if _norm(p) in forbid:
                dropped_ph += 1
            else:
                kept.append(p)
        if kept:
            clean[c] = kept
        else:
            dropped_concepts += 1
    return clean, dropped_ph, dropped_concepts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300, help="how many TRAIN concepts")
    ap.add_argument("--teacher", default="gpt-4o-mini", help="concept/phrasing model (cheap is fine)")
    ap.add_argument("--per", type=int, default=3, help="phrasings requested per concept")
    ap.add_argument("--offline", action="store_true", help="use the seed list, no API")
    ap.add_argument("--no-phrasings", action="store_true")
    args = ap.parse_args()

    eval_concepts = list(LITMUS_CONCEPTS) + EXTRA_EVAL_CONCEPTS
    # everything that must stay OUT of train: eval + exemplars
    train_exclude = eval_concepts + EXEMPLAR_CONCEPTS

    load_env()
    if args.offline:
        print(f"[offline] seed list ({len(SEED_CONCEPTS)} concepts)")
        train = _dedup_exclude(SEED_CONCEPTS, {_norm(c) for c in train_exclude})[: args.target]
        client = None
        source = "seed_list"
    else:
        from openai import OpenAI
        client = OpenAI()
        print(f"[online] generating ~{args.target} train concepts via {args.teacher}")
        train = generate_concepts(client, args.teacher, args.target, train_exclude)
        if len(train) < args.target:
            need = args.target - len(train)
            fill = _dedup_exclude(SEED_CONCEPTS,
                                  {_norm(c) for c in train_exclude + train})[:need]
            train.extend(fill)
            print(f"  topped up with {len(fill)} seed concepts")
        source = args.teacher

    phrasings = {}
    if not args.no_phrasings and not args.offline:
        print("generating phrasings...")
        phrasings = generate_phrasings(client, args.teacher, train, per=args.per)
    else:
        phrasings = {c: [c] for c in train}

    # Final leakage scrub: phrasings can collapse onto held-out eval concepts.
    phrasings, dph, dc = scrub_leakage(phrasings, eval_concepts + EXEMPLAR_CONCEPTS)
    train = [c for c in train if c in phrasings]
    if dph or dc:
        print(f"leakage scrub: dropped {dph} phrasings, {dc} concepts")

    n_pairs = sum(len(v) for v in phrasings.values())
    payload = {
        "meta": {
            "domain": "elementary physical + life science",
            "source": source,
            "n_train_concepts": len(train),
            "n_train_phrasings": n_pairs,
            "n_eval": len(eval_concepts),
            "phrasings_per_concept": args.per,
            "eval_includes_litmus_12": True,
            "note": "eval = 12 litmus + 12 extra, all held out of train. exemplar "
                    "concepts also excluded from train. phrasings only for train.",
        },
        "eval": eval_concepts,
        "eval_litmus": list(LITMUS_CONCEPTS),
        "train_concepts": train,
        "phrasings": phrasings,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    print(f"  train concepts: {len(train)}  ->  {n_pairs} (concept,phrasing) pairs")
    print(f"  eval concepts:  {len(eval_concepts)}  (incl. 12 litmus)")


if __name__ == "__main__":
    main()
