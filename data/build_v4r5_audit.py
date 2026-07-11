"""Build a deterministic broad audit slice without using benchmark failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.v4r3 import TARGETED_V4R3_ITEMS, norm_text, training_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "v4" / "gold_v4_r4.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "v4" / "v4r4_broad_accuracy_audit.json"
HISTORICAL_TARGETED_PROMPTS = {
    norm_text(training_prompt(record)) for record in TARGETED_V4R3_ITEMS
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_rank(record: dict) -> str:
    value = f"{training_prompt(record)}\n{record.get('explanation', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_records(records: list[dict], sample_size: int) -> list[dict]:
    candidates = []
    seen_prompts = set()
    for record in sorted(records, key=stable_rank):
        prompt_key = norm_text(training_prompt(record))
        if prompt_key in seen_prompts or prompt_key in HISTORICAL_TARGETED_PROMPTS:
            continue
        seen_prompts.add(prompt_key)
        candidates.append(record)

    selected = []
    selected_ids = set()
    seen_concepts = set()
    for record in candidates:
        concept_key = norm_text(record.get("concept", ""))
        if concept_key in seen_concepts:
            continue
        selected.append(record)
        selected_ids.add(id(record))
        seen_concepts.add(concept_key)
        if len(selected) >= sample_size:
            return selected

    for record in candidates:
        if id(record) in selected_ids:
            continue
        selected.append(record)
        if len(selected) >= sample_size:
            break
    return selected


def normalize(records: list[dict], source: Path) -> list[dict]:
    normalized = []
    for record in records:
        metrics = record.get("metrics", record.get("fk", {}))
        teacher = record.get("teacher", "mixed")
        normalized.append(
            {
                "model_key": "v4r4_training_audit",
                "label": "v4r4 broad training-target audit",
                "model_id": teacher,
                "tested_family": "mixed_teacher",
                "prompt": "training phrasing",
                "concept": training_prompt(record),
                "text": record["explanation"],
                "readability_pass": bool(
                    metrics.get("readability_pass_v4", True)
                ),
                "source": source.name,
                "training_metadata": {
                    "concept": record.get("concept"),
                    "teacher": teacher,
                    "rewriters": record.get("rewriters", []),
                    "judge": record.get("judge"),
                    "rewrite_iters": record.get("rewrite_iters", 0),
                    "accuracy_repairs": record.get("accuracy_repairs", 0),
                },
            }
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-size", type=int, default=40)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")

    selected = select_records(load_jsonl(args.input), args.sample_size)
    normalized = normalize(selected, args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "selection_policy": (
                    "stable broad sample, one record per concept before repeats; "
                    "historical failure-targeted prompts excluded"
                ),
                "requested_records": args.sample_size,
                "records": normalized,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(normalized)}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
