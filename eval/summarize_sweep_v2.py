"""Summarize accuracy-v2 judged decoding-sweep records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(data: dict) -> list[dict]:
    grouped = defaultdict(list)
    for record in data["records"]:
        if record.get("consensus") is None:
            raise ValueError("all sweep records must have consensus scores")
        grouped[record["model_key"]].append(record)

    rows = []
    for model_key, records in grouped.items():
        rows.append(
            {
                "model_key": model_key,
                "label": records[0]["label"],
                "n": len(records),
                "readability": sum(record["readability_pass"] for record in records),
                "clean_accuracy": sum(record["consensus"]["clean_pass"] for record in records),
                "accuracy_v2": sum(
                    record["consensus"]["accuracy_pass_v2"] for record in records
                ),
                "overall_v2": sum(record["overall_pass_v2"] for record in records),
                "gemini": sum(record["consensus"]["tiebreaker_used"] for record in records),
            }
        )
    return sorted(rows, key=lambda row: (-row["overall_v2"], -row["readability"], row["model_key"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    rows = summarize(data)
    print("| setting | readability | clean | accuracy-v2 | overall-v2 | Gemini |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['label']} | {row['readability']}/{row['n']} | "
            f"{row['clean_accuracy']}/{row['n']} | {row['accuracy_v2']}/{row['n']} | "
            f"**{row['overall_v2']}/{row['n']}** | {row['gemini']}/{row['n']} |"
        )
    if args.out:
        args.out.write_text(json.dumps({"settings": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
