"""Generate a reproducible high-end frontier comparison on the fixed litmus set."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from litmus.concepts import CONCEPTS, PROMPT_TEMPLATE, build_prompt
from litmus.env import make_client
from litmus.fk_score import score_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MODELS = [
    "openai-group/gpt-5.6-sol",
    "claude-group/claude-opus-4-7",
    "gemini-group/gemini-3.1-pro",
]
DEFAULT_OUT = Path(__file__).resolve().parent / "frontier_v2_outputs.json"
MODEL_LABELS = {
    "openai-group/gpt-5.6-sol": "GPT-5.6 SOL",
    "claude-group/claude-fable-5": "Claude Fable 5",
    "claude-group/claude-opus-4-7": "Claude Opus 4.7",
    "gemini-group/gemini-3.1-pro": "Gemini 3.1 Pro",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))


def model_family(model: str) -> str:
    lowered = model.lower()
    if "openai" in lowered or "gpt" in lowered:
        return "openai"
    if "claude" in lowered or "anthropic" in lowered:
        return "anthropic"
    if "gemini" in lowered or "google" in lowered:
        return "google"
    return "other"


def model_key(model: str) -> str:
    return "frontier_v2_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def normalized_record(model: str, concept: str, text: str, out_name: str) -> dict:
    readability = score_text(text)
    return {
        "model_key": model_key(model),
        "label": MODEL_LABELS.get(model, model),
        "model_id": model,
        "tested_family": model_family(model),
        "prompt": "full grade-3 prompt",
        "concept": concept,
        "text": text,
        "readability_pass": bool(readability["readability_pass_v4"]),
        "source": out_name,
        "readability": {
            "whole_passage_fk": readability["whole_passage_fk"],
            "whole_passage_ari": readability["whole_passage_ari"],
            "fk_stdev": readability["fk_stdev"],
            "max_fk": readability["max_fk"],
        },
    }


def preflight_models(client, models: list[str]) -> None:
    available = {item.id for item in client.models.list().data}
    missing = [model for model in models if model not in available]
    if missing:
        raise RuntimeError(f"gateway preflight missing model(s): {', '.join(missing)}")
    print(f"[preflight] found {len(models)} frontier model slugs")


def generate_one(client, model: str, concept: str, temperature: float,
                 max_retries: int) -> str:
    delay = 2.0
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": build_prompt(concept)}],
            )
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ValueError("empty model response")
            return text.strip()
        except Exception as error:  # noqa: BLE001 - bounded gateway retries
            last_error = error
            if attempt >= max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"{model} failed for {concept}: {last_error}") from last_error


def load_existing(path: Path, resume: bool) -> dict | None:
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(f"{path} exists; pass --resume to reuse it")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "frontier_v2_generation_v1":
        raise ValueError("existing file has a different schema_version")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    concepts = CONCEPTS[: args.limit] if args.limit else CONCEPTS
    if not models or args.temperature < 0:
        parser.error("models must be non-empty and temperature must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.concurrency < 1 or args.max_retries < 1:
        parser.error("--concurrency and --max-retries must be positive")

    existing = load_existing(args.out, args.resume)
    records = existing.get("records", []) if existing else []
    completed = {(record["model_id"], record["concept"]): record for record in records}
    tasks = [
        (model, concept)
        for model in models
        for concept in concepts
        if (model, concept) not in completed
    ]
    if args.dry_run:
        print(f"models: {len(models)}")
        print(f"concepts: {len(concepts)}")
        print(f"completed outputs: {len(completed)}")
        print(f"missing generation calls: {len(tasks)}")
        return

    client = make_client()
    if not args.skip_preflight:
        preflight_models(client, models)
    data = {
        "schema_version": "frontier_v2_generation_v1",
        "created_at": existing.get("created_at") if existing else utc_now(),
        "updated_at": utc_now(),
        "models": models,
        "temperature": args.temperature,
        "prompt_template": PROMPT_TEMPLATE,
        "records": list(completed.values()),
    }
    atomic_write(args.out, data)

    def work(task):
        model, concept = task
        text = generate_one(client, model, concept, args.temperature, args.max_retries)
        return normalized_record(model, concept, text, args.out.name)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(work, task): task for task in tasks}
        errors = []
        for index, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - report all provider failures
                model, concept = futures[future]
                errors.append(f"{model} / {concept}: {error}")
                print(f"[{index}/{len(tasks)}] ERROR | {model} | {concept}", flush=True)
                continue
            completed[(record["model_id"], record["concept"])] = record
            data["records"] = [
                completed[(model, concept)]
                for model in models
                for concept in concepts
                if (model, concept) in completed
            ]
            data["updated_at"] = utc_now()
            atomic_write(args.out, data)
            print(
                f"[{index}/{len(tasks)}] read={record['readability_pass']} | "
                f"{record['label']} | {record['concept']}",
                flush=True,
            )

    if errors:
        raise RuntimeError("frontier generation failures:\n" + "\n".join(errors))

    if len(data["records"]) != len(models) * len(concepts):
        raise RuntimeError("frontier generation ended with an incomplete output matrix")
    print(f"complete: {len(data['records'])} outputs")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
