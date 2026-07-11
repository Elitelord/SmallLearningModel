"""Accuracy-v2 rubric, validation, and multi-judge consensus helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

RUBRIC_VERSION = "accuracy_v2"
SCHEMA_VERSION = 1

GPT_JUDGE = "openai-group/gpt-5.4"
CLAUDE_JUDGE = "claude-group/claude-opus-4-7"
GEMINI_TIEBREAKER = "gemini-group/gemini-3.1-pro"

FACTUALITY_MIN = 0
FACTUALITY_MAX = 3
MECHANISM_MIN = 0
MECHANISM_MAX = 2


ACCURACY_V2_RUBRIC = """Judge this elementary-science explanation on TWO independent axes.

FACTUALITY (0-3)
3 = No substantive factual errors or misleading claims.
2 = The core answer is correct, with only a minor/local imprecision that does not change the core mechanism.
1 = A major misleading or false claim affects the causal explanation, though some relevant content is correct.
0 = The central answer or mechanism is fundamentally wrong, contradictory, or irrelevant.

MECHANISM (0-2)
2 = Clearly conveys the core cause-and-effect chain at a level a 7-year-old can understand.
1 = Gives a partial causal link or mostly names/restates the phenomenon without a complete core mechanism.
0 = The mechanism is absent, circular, or itself wrong.

Audience calibration: do not demand adult terminology or textbook depth. Simple language and omitted advanced detail are not errors. Judge only scientific correctness and whether the child receives the real core HOW/WHY.

Error-list rules:
- factuality 3: errors must be an empty list.
- factuality 2: list only minor errors.
- factuality 1 or 0: include at least one major error.
- Each error must quote or closely identify the problematic claim and provide a concise correction.
"""


def build_judge_prompt_v2(concept: str, explanation: str) -> str:
    return (
        f"{ACCURACY_V2_RUBRIC}\n"
        f"CONCEPT: {concept}\n\n"
        f"EXPLANATION TO JUDGE:\n{explanation}\n\n"
        "Return only a JSON object with this exact shape:\n"
        '{"factuality": 0, "mechanism": 0, '
        '"errors": [{"severity": "minor|major", "claim": "...", '
        '"correction": "..."}], "justification": "one concise sentence"}'
    )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bounded_int(value, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def validate_judgment(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("judgment must be a JSON object")

    factuality = _bounded_int(
        data.get("factuality"), FACTUALITY_MIN, FACTUALITY_MAX, "factuality"
    )
    mechanism = _bounded_int(
        data.get("mechanism"), MECHANISM_MIN, MECHANISM_MAX, "mechanism"
    )
    errors = data.get("errors")
    if not isinstance(errors, list):
        raise ValueError("errors must be a list")

    normalized_errors = []
    for error in errors:
        if not isinstance(error, dict):
            raise ValueError("each error must be an object")
        severity = error.get("severity")
        claim = error.get("claim")
        correction = error.get("correction")
        if severity not in {"minor", "major"}:
            raise ValueError("error severity must be minor or major")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("error claim must be non-empty")
        if not isinstance(correction, str) or not correction.strip():
            raise ValueError("error correction must be non-empty")
        normalized_errors.append(
            {
                "severity": severity,
                "claim": claim.strip(),
                "correction": correction.strip(),
            }
        )

    if factuality == 3 and normalized_errors:
        raise ValueError("factuality 3 requires an empty errors list")
    if factuality < 3 and not normalized_errors:
        raise ValueError("factuality below 3 requires at least one error")
    if factuality == 2 and any(e["severity"] != "minor" for e in normalized_errors):
        raise ValueError("factuality 2 may contain only minor errors")
    if factuality <= 1 and not any(e["severity"] == "major" for e in normalized_errors):
        raise ValueError("factuality 0 or 1 requires a major error")

    justification = data.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise ValueError("justification must be a non-empty string")

    return {
        "factuality": factuality,
        "mechanism": mechanism,
        "errors": normalized_errors,
        "justification": justification.strip(),
    }


def clean_pass(factuality: int, mechanism: int) -> bool:
    return factuality == 3 and mechanism == 2


def accuracy_pass_v2(factuality: int, mechanism: int) -> bool:
    return factuality >= 2 and mechanism == 2


def judges_disagree(first: dict, second: dict) -> bool:
    return (
        first["factuality"] != second["factuality"]
        or first["mechanism"] != second["mechanism"]
    )


def _median(values: Iterable[int]) -> int:
    ordered = sorted(values)
    if len(ordered) % 2 == 0:
        raise ValueError("consensus median requires an odd number of scores")
    return ordered[len(ordered) // 2]


def build_consensus(first: dict, second: dict, tiebreaker: dict | None = None) -> dict:
    disagreement = judges_disagree(first, second)
    if disagreement and tiebreaker is None:
        raise ValueError("a tiebreaker judgment is required when primary judges disagree")
    if not disagreement and tiebreaker is not None:
        raise ValueError("tiebreaker must not be used when primary judges agree")

    judgments = [first, second] if not disagreement else [first, second, tiebreaker]
    if disagreement:
        factuality = _median(j["factuality"] for j in judgments)
        mechanism = _median(j["mechanism"] for j in judgments)
        method = "three_judge_axis_median"
    else:
        factuality = first["factuality"]
        mechanism = first["mechanism"]
        method = "primary_exact_agreement"

    return {
        "factuality": factuality,
        "mechanism": mechanism,
        "total": factuality + mechanism,
        "clean_pass": clean_pass(factuality, mechanism),
        "accuracy_pass_v2": accuracy_pass_v2(factuality, mechanism),
        "primary_disagreement": disagreement,
        "tiebreaker_used": disagreement,
        "method": method,
    }


def linear_weighted_kappa(first_scores: list[int], second_scores: list[int], maximum: int) -> float:
    if len(first_scores) != len(second_scores) or not first_scores:
        raise ValueError("score lists must be non-empty and the same length")
    if maximum <= 0:
        raise ValueError("maximum must be positive")

    size = maximum + 1
    observed = [[0.0 for _ in range(size)] for _ in range(size)]
    first_hist = [0.0 for _ in range(size)]
    second_hist = [0.0 for _ in range(size)]
    for first, second in zip(first_scores, second_scores):
        _bounded_int(first, 0, maximum, "first score")
        _bounded_int(second, 0, maximum, "second score")
        observed[first][second] += 1.0
        first_hist[first] += 1.0
        second_hist[second] += 1.0

    count = float(len(first_scores))
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for first in range(size):
        for second in range(size):
            weight = abs(first - second) / maximum
            observed_disagreement += weight * observed[first][second] / count
            expected_disagreement += weight * (first_hist[first] / count) * (second_hist[second] / count)
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - observed_disagreement / expected_disagreement
