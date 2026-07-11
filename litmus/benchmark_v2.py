"""Normalize saved baseline and tuned outputs for the accuracy-v2 benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from litmus.concepts import CONCEPTS
from litmus.fk_score import score_text
from litmus.score_all import load_all_outputs

REPO_ROOT = Path(__file__).resolve().parent.parent

BASELINE_MODELS = {
    "gpt": {
        "label": "GPT-4o",
        "prompt": "full grade-3 prompt",
        "family": "openai",
    },
    "claude_browser": {
        "label": "Claude (browser)",
        "prompt": "full grade-3 prompt",
        "family": "anthropic",
    },
    "gemini": {
        "label": "Gemini (browser)",
        "prompt": "full grade-3 prompt",
        "family": "google",
    },
    "qwen_4b": {
        "label": "Qwen3-4B (base)",
        "prompt": "full grade-3 prompt",
        "family": "qwen",
    },
    "qwen_0.6b": {
        "label": "Qwen3-0.6B (reference)",
        "prompt": "full grade-3 prompt",
        "family": "qwen",
    },
}

TUNED_MODELS = {
    "v4r2": {
        "label": "Qwen3-4B + v2 tune (v4r2)",
        "path": REPO_ROOT / "base_vs_tuned_results.json",
    },
    "v4r3": {
        "label": "Qwen3-4B + v3 tune (v4r3)",
        "path": REPO_ROOT / "base_vs_tuned_v4r3_all24_readability.json",
    },
    "v4r4": {
        "label": "Qwen3-4B + v4 tune (v4r4)",
        "path": REPO_ROOT / "base_vs_tuned_v4r4_all24_readability.json",
    },
}

MODEL_ORDER = [*BASELINE_MODELS, *TUNED_MODELS]


def _baseline_records() -> list[dict]:
    saved = load_all_outputs()
    records = []
    for model_key, metadata in BASELINE_MODELS.items():
        if model_key not in saved:
            raise ValueError(f"missing baseline outputs for {model_key}")
        info = saved[model_key]
        for concept in CONCEPTS:
            text = info["outputs"].get(concept)
            if not text or not text.strip():
                raise ValueError(f"missing output for {model_key}: {concept}")
            readability = score_text(text)
            records.append(
                {
                    "model_key": model_key,
                    "label": metadata["label"],
                    "model_id": info["model_id"],
                    "tested_family": metadata["family"],
                    "prompt": metadata["prompt"],
                    "concept": concept,
                    "text": text,
                    "readability_pass": bool(readability["readability_pass_v4"]),
                    "source": f"litmus outputs ({model_key})",
                }
            )
    return records


def _tuned_records() -> list[dict]:
    records = []
    for model_key, metadata in TUNED_MODELS.items():
        path = metadata["path"]
        if not path.exists():
            raise ValueError(f"missing tuned results file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = {row["concept"]: row for row in data["rows"]}
        for concept in CONCEPTS:
            if concept not in rows:
                raise ValueError(f"missing output for {model_key}: {concept}")
            row = rows[concept]
            text = row.get("tuned_text")
            if not text or not text.strip():
                raise ValueError(f"empty tuned output for {model_key}: {concept}")
            records.append(
                {
                    "model_key": model_key,
                    "label": metadata["label"],
                    "model_id": data.get("model", "Qwen/Qwen3-4B"),
                    "tested_family": "qwen",
                    "prompt": "bare Explain:",
                    "concept": concept,
                    "text": text,
                    "readability_pass": bool(row["tuned"]["readability_pass"]),
                    "source": path.name,
                }
            )
    return records


def load_benchmark_records() -> list[dict]:
    records = _baseline_records() + _tuned_records()
    expected = len(MODEL_ORDER) * len(CONCEPTS)
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, found {len(records)}")
    identities = {(record["model_key"], record["concept"]) for record in records}
    if len(identities) != expected:
        raise ValueError("duplicate model/concept records found")
    return records
