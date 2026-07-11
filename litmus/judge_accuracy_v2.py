"""Run the resumable accuracy-v2 multi-judge benchmark.

Primary judges score every saved output. Gemini is called only when the primary
judges differ on factuality or mechanism. Results are checkpointed after every
successful call so an interrupted paid run can resume without duplicate work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from litmus.accuracy_v2 import (
    CLAUDE_JUDGE,
    GEMINI_TIEBREAKER,
    GPT_JUDGE,
    RUBRIC_VERSION,
    SCHEMA_VERSION,
    build_consensus,
    build_judge_prompt_v2,
    judges_disagree,
    text_sha256,
    validate_judgment,
)
from litmus.benchmark_v2 import load_benchmark_records
from litmus.env import make_client

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_OUT = Path(__file__).resolve().parent / "accuracy_v2_scores.json"
REQUIRED_RECORD_FIELDS = {
    "model_key",
    "label",
    "model_id",
    "tested_family",
    "prompt",
    "concept",
    "text",
    "readability_pass",
    "source",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def judge_family(model: str) -> str:
    lowered = model.lower()
    if "openai" in lowered or "gpt" in lowered:
        return "openai"
    if "claude" in lowered or "anthropic" in lowered:
        return "anthropic"
    if "gemini" in lowered or "google" in lowered:
        return "google"
    return "other"


def atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_existing(path: Path, resume: bool) -> dict | None:
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(f"{path} exists; pass --resume to reuse it")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("existing score file has a different schema_version")
    if data.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError("existing score file has a different rubric_version")
    return data


def load_input_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list) or not records:
        raise ValueError("custom input must contain a non-empty records list")
    normalized = []
    identities = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        missing = REQUIRED_RECORD_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record {index} missing fields: {', '.join(sorted(missing))}")
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError(f"record {index} has empty text")
        identity = (record["model_key"], record["concept"])
        if identity in identities:
            raise ValueError(f"duplicate custom record: {identity}")
        identities.add(identity)
        normalized.append({key: record[key] for key in record})
    return normalized


def prepare_output(records: list[dict], existing: dict | None, judges: dict) -> dict:
    existing_lookup = {}
    if existing:
        existing_lookup = {
            (record["model_key"], record["concept"]): record
            for record in existing.get("records", [])
        }

    prepared = []
    for source in records:
        identity = (source["model_key"], source["concept"])
        digest = text_sha256(source["text"])
        old = existing_lookup.get(identity)
        saved_judgments = {}
        if old and old.get("text_sha256") == digest:
            for model, judgment in old.get("judgments", {}).items():
                try:
                    saved_judgments[model] = validate_judgment(judgment)
                except ValueError:
                    continue
        prepared.append(
            {
                **source,
                "text_sha256": digest,
                "judgments": saved_judgments,
                "consensus": None,
            }
        )

    created_at = existing.get("created_at") if existing else utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "created_at": created_at,
        "updated_at": utc_now(),
        "judges": judges,
        "records": prepared,
    }


def preflight_models(client, model_ids: list[str]) -> None:
    response = client.models.list()
    available = {model.id for model in response.data}
    missing = [model for model in model_ids if model not in available]
    if missing:
        raise RuntimeError(f"gateway preflight missing model(s): {', '.join(missing)}")
    print(f"[preflight] found {len(model_ids)} required model slugs")


def judge_one(client, model: str, record: dict, max_retries: int) -> dict:
    prompt = build_judge_prompt_v2(record["concept"], record["text"])
    delay = 2.0
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("empty judge response")
            judgment = validate_judgment(json.loads(content))
            judgment["judged_at"] = utc_now()
            return judgment
        except Exception as error:  # noqa: BLE001 - paid calls need bounded retries
            last_error = error
            if attempt >= max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(
        f"{model} failed for {record['model_key']} / {record['concept']}: {last_error}"
    ) from last_error


def missing_tasks(data: dict, models: list[str]) -> list[tuple[int, str]]:
    tasks = []
    for index, record in enumerate(data["records"]):
        for model in models:
            if model not in record["judgments"]:
                tasks.append((index, model))
    return tasks


def run_tasks(client, data: dict, tasks: list[tuple[int, str]], path: Path,
              concurrency: int, max_retries: int) -> None:
    if not tasks:
        return

    def work(task):
        index, model = task
        return index, model, judge_one(client, model, data["records"][index], max_retries)

    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(work, task): task for task in tasks}
        for future in as_completed(futures):
            index, model, judgment = future.result()
            data["records"][index]["judgments"][model] = judgment
            data["updated_at"] = utc_now()
            atomic_write(path, data)
            completed += 1
            record = data["records"][index]
            print(
                f"[{completed}/{len(tasks)}] {model} "
                f"F{judgment['factuality']}/M{judgment['mechanism']} | "
                f"{record['model_key']} | {record['concept']}",
                flush=True,
            )


def tiebreaker_tasks(data: dict, primary_models: list[str], tiebreaker: str) -> list[tuple[int, str]]:
    tasks = []
    for index, record in enumerate(data["records"]):
        first = record["judgments"].get(primary_models[0])
        second = record["judgments"].get(primary_models[1])
        if first and second and judges_disagree(first, second) and tiebreaker not in record["judgments"]:
            tasks.append((index, tiebreaker))
    return tasks


def finalize_consensus(data: dict, primary_models: list[str], tiebreaker: str) -> None:
    unresolved = []
    for record in data["records"]:
        first = record["judgments"].get(primary_models[0])
        second = record["judgments"].get(primary_models[1])
        if not first or not second:
            unresolved.append(f"{record['model_key']} / {record['concept']}: missing primary")
            continue
        disagreement = judges_disagree(first, second)
        third = record["judgments"].get(tiebreaker) if disagreement else None
        if disagreement and not third:
            unresolved.append(f"{record['model_key']} / {record['concept']}: missing tiebreaker")
            continue
        record["consensus"] = build_consensus(first, second, third)
        record["overall_pass_v2"] = bool(
            record["readability_pass"] and record["consensus"]["accuracy_pass_v2"]
        )
        record["judge_family_relationships"] = {
            model: judge_family(model) == record["tested_family"]
            for model in record["judgments"]
        }
    if unresolved:
        raise RuntimeError("unresolved judgments:\n" + "\n".join(unresolved))


def print_dry_run(data: dict, primary_models: list[str], tiebreaker: str) -> None:
    primary_missing = len(missing_tasks(data, primary_models))
    records = len(data["records"])
    potential_tiebreakers = 0
    for record in data["records"]:
        if tiebreaker in record["judgments"]:
            continue
        first = record["judgments"].get(primary_models[0])
        second = record["judgments"].get(primary_models[1])
        if first is None or second is None or judges_disagree(first, second):
            potential_tiebreakers += 1
    print(f"records: {records}")
    print(f"primary judges: {', '.join(primary_models)}")
    print(f"tiebreaker: {tiebreaker}")
    print(f"missing required primary calls: {primary_missing}")
    print(f"maximum additional tiebreaker calls: {potential_tiebreakers}")
    print(f"maximum total additional calls: {primary_missing + potential_tiebreakers}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-judge", default=GPT_JUDGE)
    parser.add_argument("--claude-judge", default=CLAUDE_JUDGE)
    parser.add_argument("--tiebreaker", default=GEMINI_TIEBREAKER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input", type=Path, help="custom normalized records JSON")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1 or args.max_retries < 1:
        parser.error("--concurrency and --max-retries must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    records = load_input_records(args.input) if args.input else load_benchmark_records()
    if args.limit is not None:
        records = records[: args.limit]
    primary_models = [args.gpt_judge, args.claude_judge]
    if len(set([*primary_models, args.tiebreaker])) != 3:
        parser.error("all three judge model slugs must be distinct")

    judges = {"primary": primary_models, "tiebreaker": args.tiebreaker}
    existing = load_existing(args.out, args.resume)
    data = prepare_output(records, existing, judges)

    if args.dry_run:
        print_dry_run(data, primary_models, args.tiebreaker)
        return

    client = make_client()
    if not args.skip_preflight:
        preflight_models(client, [*primary_models, args.tiebreaker])

    atomic_write(args.out, data)
    run_tasks(
        client,
        data,
        missing_tasks(data, primary_models),
        args.out,
        args.concurrency,
        args.max_retries,
    )
    run_tasks(
        client,
        data,
        tiebreaker_tasks(data, primary_models, args.tiebreaker),
        args.out,
        args.concurrency,
        args.max_retries,
    )
    finalize_consensus(data, primary_models, args.tiebreaker)
    data["updated_at"] = utc_now()
    atomic_write(args.out, data)

    tiebreaks = sum(record["consensus"]["tiebreaker_used"] for record in data["records"])
    print(f"complete: {len(data['records'])} records, {tiebreaks} Gemini tiebreakers")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
