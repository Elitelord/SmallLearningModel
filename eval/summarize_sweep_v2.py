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
        concepts = [record["concept"] for record in records]
        if len(concepts) != len(set(concepts)):
            raise ValueError(f"duplicate concepts for {model_key}")
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


def passes_gate(rows: list[dict], expected_n: int, minimum_overall: int) -> bool:
    if expected_n < 1 or minimum_overall < 0 or minimum_overall > expected_n:
        raise ValueError("invalid overall gate")
    return bool(
        rows
        and all(row["n"] == expected_n for row in rows)
        and any(row["overall_v2"] >= minimum_overall for row in rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-n", type=int)
    parser.add_argument("--require-overall", type=int)
    args = parser.parse_args()

    if (args.require_n is None) != (args.require_overall is None):
        parser.error("--require-n and --require-overall must be used together")

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
    if args.require_n is not None:
        if not passes_gate(rows, args.require_n, args.require_overall):
            raise SystemExit(
                f"overall gate failed: need at least {args.require_overall}/{args.require_n} "
                "from complete unique-concept settings"
            )
        print(f"\noverall gate passed: >= {args.require_overall}/{args.require_n}")


if __name__ == "__main__":
    main()
