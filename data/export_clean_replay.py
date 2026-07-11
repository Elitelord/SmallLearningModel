"""Export consensus-clean audited targets back to training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "v4" / "v4r4_broad_accuracy_audit.scores.json"
DEFAULT_OUT = REPO_ROOT / "data" / "v4" / "v4r4_broad_replay_clean.jsonl"


def clean_replay_records(data: dict) -> list[dict]:
    replay = []
    for record in data["records"]:
        consensus = record.get("consensus")
        if not consensus or not consensus.get("clean_pass"):
            continue
        metadata = record.get("training_metadata", {})
        replay.append(
            {
                "concept": metadata.get("concept") or record["concept"],
                "phrasing": record["concept"],
                "explanation": record["text"],
                "accuracy": {
                    "score": 2,
                    "rubric": data.get("rubric_version", "accuracy_v2"),
                    "gate": "clean",
                    "judgments": record["judgments"],
                    "consensus": consensus,
                    "judges": list(record["judgments"]),
                },
                "teacher": metadata.get("teacher", "v4r4-replay"),
                "rewriters": metadata.get("rewriters", []),
                "judge": list(record["judgments"]),
                "rewrite_iters": metadata.get("rewrite_iters", 0),
                "accuracy_repairs": metadata.get("accuracy_repairs", 0),
                "seed_source": record.get("source"),
            }
        )
    return replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    records = clean_replay_records(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(records)} consensus-clean records)")


if __name__ == "__main__":
    main()
