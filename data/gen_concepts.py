"""Part A.1 - generate the concept list via the teacher, dedup, split.

Produces data/concepts.json:
    {
      "meta": {...},
      "eval":  [...the 12 held-out litmus concepts...],
      "train": [...generated concepts, litmus concepts EXCLUDED...]
    }

Guardrails baked in:
  - NARROW DOMAIN: elementary physical + life science only. A 0.6B student learns
    form faster than facts; a wide domain yields fluent-but-wrong grade-3 text.
  - NO LEAKAGE: the 12 litmus concepts are the eval set and are removed from train
    by normalized-string match (see _norm).

Usage:
    .venv\\Scripts\\python -m data.gen_concepts --target 60          # smoke: ~60
    .venv\\Scripts\\python -m data.gen_concepts --target 300         # real day-3 run
    .venv\\Scripts\\python -m data.gen_concepts --target 60 --offline # no API, seed list
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

# A small deterministic seed list for --offline runs (and as a fallback if the
# teacher returns too few). These are ordinary elementary-science questions and
# are NOT among the 12 litmus concepts.
SEED_CONCEPTS = [
    "Why do we sweat when it is hot?",
    "How does a seed grow into a plant?",
    "Why do we need to eat food?",
    "Why does ice float on water?",
    "Why do leaves change color in fall?",
    "What is a shadow?",
    "Why do we get goosebumps?",
    "How do bees help flowers?",
    "Why does the ocean have waves?",
    "Why do we yawn?",
    "How does a bird fly?",
    "Why is grass green?",
    "What makes thunder so loud?",
    "Why do we have to sleep?",
    "How do our ears hear sound?",
    "Why does bread rise when it bakes?",
    "Why do cats have whiskers?",
    "How does a caterpillar turn into a butterfly?",
    "Why do we shiver when we are cold?",
    "What makes a balloon float in the air?",
    "Why do stars twinkle at night?",
    "How does soap get our hands clean?",
    "Why do we have eyebrows?",
    "Why does a ball roll downhill?",
    "How do plants drink water?",
    "Why do we get hungry?",
    "What makes popcorn pop?",
    "Why do dogs pant?",
    "How does a boat stay on top of the water?",
    "Why do we have bones?",
    "Why does the wind blow?",
    "How do spiders make their webs?",
    "Why do apples turn brown after we cut them?",
    "Why do we blink our eyes?",
    "How does a magnet pick up metal?",
    "Why does milk go bad if we leave it out?",
    "Why do we have teeth of different shapes?",
    "How does a frog catch a fly?",
    "Why does snow melt in our hands?",
    "Why do some things sink and others float?",
]


def _norm(s: str) -> str:
    """Normalize a concept for dedup and leakage checks."""
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


def generate_concepts(target: int, batch: int = 60, temperature: float = 1.0):
    """Ask the teacher for ~target distinct concepts, in batches for diversity."""
    load_env()
    from openai import OpenAI

    client = OpenAI()
    litmus_norms = {_norm(c) for c in LITMUS_CONCEPTS}
    collected = []
    seen = set(litmus_norms)
    rounds = 0
    while len(collected) < target and rounds < 12:
        rounds += 1
        avoid = LITMUS_CONCEPTS + collected[-40:]
        prompt = (
            f"List {batch} distinct science questions about {DOMAIN}.\n"
            "Each must be a short, natural question a young child would ask, ending in '?'.\n"
            "Vary the topics widely; do not cluster on one theme.\n"
            "Do NOT repeat or paraphrase any of these already-used questions:\n"
            + "\n".join(f"- {a}" for a in avoid)
            + '\n\nReturn ONLY a JSON object: {"concepts": ["...", "..."]}'
        )
        resp = client.chat.completions.create(
            model=TEACHER_MODEL,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.choices[0].message.content)
        fresh = _dedup_exclude(data.get("concepts", []), seen)
        for c in fresh:
            seen.add(_norm(c))
        collected.extend(fresh)
        print(f"  round {rounds}: +{len(fresh)} -> {len(collected)}/{target}")
        if not fresh:
            break
    return collected[:target]


TEACHER_MODEL = "gpt-4o-mini"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=60, help="how many TRAIN concepts to gather")
    ap.add_argument("--teacher", default="gpt-4o-mini")
    ap.add_argument("--offline", action="store_true", help="use the seed list, no API calls")
    args = ap.parse_args()

    global TEACHER_MODEL
    TEACHER_MODEL = args.teacher

    litmus_norms = {_norm(c) for c in LITMUS_CONCEPTS}

    if args.offline:
        print(f"[offline] using {len(SEED_CONCEPTS)} seed concepts")
        train = _dedup_exclude(SEED_CONCEPTS, litmus_norms)[: args.target]
        source = "seed_list"
    else:
        print(f"[online] generating ~{args.target} concepts via {TEACHER_MODEL}")
        train = generate_concepts(args.target)
        if len(train) < args.target:
            # top up from the seed list so smoke runs are never starved
            need = args.target - len(train)
            fill = _dedup_exclude(SEED_CONCEPTS, litmus_norms | {_norm(c) for c in train})[:need]
            train.extend(fill)
            print(f"  topped up with {len(fill)} seed concepts")
        source = TEACHER_MODEL

    payload = {
        "meta": {
            "domain": "elementary physical + life science",
            "source": source,
            "n_train": len(train),
            "n_eval": len(LITMUS_CONCEPTS),
            "eval_is_held_out": True,
            "note": "eval = the 12 litmus concepts, held out of train entirely (no leakage).",
        },
        "eval": list(LITMUS_CONCEPTS),
        "train": train,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}: {len(train)} train + {len(LITMUS_CONCEPTS)} eval concepts")


if __name__ == "__main__":
    main()
