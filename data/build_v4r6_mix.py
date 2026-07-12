"""Build a deterministic clean r2/r4/r5 mixture for the v4r6 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.audit_v4r3 import audit, passes_accuracy_gate
from data.build_v4r5_audit import HISTORICAL_TARGETED_PROMPTS
from data.v4r3 import norm_text, target_config, training_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_R2 = REPO_ROOT / "data" / "v4" / "v4r2_tight_replay_clean.jsonl"
DEFAULT_R4 = REPO_ROOT / "data" / "v4" / "v4r4_tight_replay_clean.jsonl"
DEFAULT_R5 = REPO_ROOT / "data" / "v4" / "gold_v4_r5.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "v4" / "gold_v4_r6.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_rank(record: dict) -> str:
    value = f"{training_prompt(record)}\n{record.get('explanation', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def reserved_prompt_keys() -> set[str]:
    concepts = json.loads(
        (REPO_ROOT / "data" / "concepts.json").read_text(encoding="utf-8")
    )
    return {
        norm_text(prompt)
        for key in ("eval", "calibration_v4r5", "blind_v4r5")
        for prompt in concepts.get(key, [])
    }


def select_source(
    records: list[dict], source_name: str, count: int, used_prompts: set[str]
) -> list[dict]:
    reserved = reserved_prompt_keys()
    selected = []
    for record in sorted(records, key=stable_rank):
        prompt_key = norm_text(training_prompt(record))
        concept_key = norm_text(record.get("concept", ""))
        if (
            prompt_key in used_prompts
            or prompt_key in reserved
            or concept_key in reserved
            or prompt_key in HISTORICAL_TARGETED_PROMPTS
            or not passes_accuracy_gate(record, "clean-v2")
        ):
            continue
        copied = dict(record)
        copied["mixture_source"] = source_name
        selected.append(copied)
        used_prompts.add(prompt_key)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"{source_name} supplied only {len(selected)}/{count} eligible unique records"
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2", type=Path, default=DEFAULT_R2)
    parser.add_argument("--r4", type=Path, default=DEFAULT_R4)
    parser.add_argument("--r5", type=Path, default=DEFAULT_R5)
    parser.add_argument("--r2-count", type=int, default=98)
    parser.add_argument("--r4-count", type=int, default=102)
    parser.add_argument("--r5-count", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if min(args.r2_count, args.r4_count, args.r5_count) < 0:
        parser.error("source counts must be non-negative")

    used_prompts: set[str] = set()
    records = []
    for path, source_name, count in (
        (args.r2, "v4r2_accuracy_anchor", args.r2_count),
        (args.r4, "v4r4_readability_replay", args.r4_count),
        (args.r5, "v4r5_clean_target", args.r5_count),
    ):
        records.extend(select_source(load_jsonl(path), source_name, count, used_prompts))

    write_jsonl(args.out, records)
    summary = audit(
        args.out,
        target_config(),
        min_sentences=4,
        max_sentences=6,
        accuracy_gate="clean-v2",
        forbid_targeted_v4r3=True,
    )
    if not summary["passed"]:
        args.out.unlink(missing_ok=True)
        raise RuntimeError(
            f"constructed mixture failed audit with {summary['failure_count']} failures"
        )

    stats = {
        "records": len(records),
        "unique_prompts": len(used_prompts),
        "dataset_sha256": file_sha256(args.out),
        "sources": {
            "v4r2_accuracy_anchor": {
                "records": args.r2_count,
                "path": portable_path(args.r2),
                "sha256": file_sha256(args.r2),
            },
            "v4r4_readability_replay": {
                "records": args.r4_count,
                "path": portable_path(args.r4),
                "sha256": file_sha256(args.r4),
            },
            "v4r5_clean_target": {
                "records": args.r5_count,
                "path": portable_path(args.r5),
                "sha256": file_sha256(args.r5),
            },
        },
        "accuracy_gate": "clean-v2",
        "readability_target": target_config(),
        "historical_targeted_prompts_excluded": True,
        "reserved_prompts_excluded": True,
    }
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
