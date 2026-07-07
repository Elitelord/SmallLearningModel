"""Accuracy scoring — the OTHER half of the spec.

A passage can sail through FK and still be wrong or vacuous. Accuracy is scored
0/1/2 against the mechanism-demanding rubric, and accuracy_pass = (score == 2).

    2 = correct AND explains the real mechanism/cause
    1 = correct but only a surface definition, no mechanism
    0 = contains a factual error or misleading oversimplification

Two ways to produce scores:
  1. MANUAL (recommended, fastest + most trustworthy for ~48 outputs): fill in
     litmus/accuracy_scores.json by hand using the rubric.
  2. LLM JUDGE: judge_with_openai(). Per MT-Bench, the judge must NOT be one of
     the tested models (avoid self-enhancement bias). Since the only API key
     here is OpenAI and GPT is under test, the judge would bias the GPT column —
     so this path prints a warning and should only be used for models the judge
     family is not itself producing.

Schema of litmus/accuracy_scores.json:
    { "<model_key>": { "<concept>": {"score": 0|1|2, "note": "..."} } }
"""

import json
from pathlib import Path

ACCURACY_RUBRIC = (
    "Score the explanation's ACCURACY on this 0/1/2 scale:\n"
    "  2 = scientifically correct AND explains the real mechanism/cause "
    "(the actual HOW/WHY), not just a definition.\n"
    "  1 = correct but only a surface definition or restatement; the mechanism "
    "is missing.\n"
    "  0 = contains a factual error or a misleading oversimplification that "
    "misrepresents the mechanism.\n"
    "Judge ONLY accuracy/mechanism here — ignore reading level."
)

SCORES_PATH = Path(__file__).resolve().parent / "accuracy_scores.json"


def load_accuracy(path: Path | None = None) -> dict:
    path = path or SCORES_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def accuracy_pass(score) -> bool:
    return score == 2


def build_judge_prompt(concept: str, output: str) -> str:
    return (
        f"{ACCURACY_RUBRIC}\n\n"
        f"CONCEPT (what a child asked): {concept}\n\n"
        f"EXPLANATION TO JUDGE:\n{output}\n\n"
        "Respond with a JSON object: "
        '{"score": 0|1|2, "justification": "one line"}'
    )


def judge_with_openai(
    outputs_by_model: dict, judge_model: str = "gpt-4o", tested_keys=("gpt",)
) -> dict:
    """Optionally auto-score accuracy with an OpenAI judge.

    outputs_by_model: {model_key: {concept: output_text}}
    Returns the same accuracy_scores.json structure.
    """
    from litmus.env import load_env

    load_env()
    from openai import OpenAI

    client = OpenAI()
    print(
        "WARNING: LLM-judge accuracy is a convenience path. The judge model "
        f"({judge_model}) is in the GPT family; scores for tested GPT models "
        "carry self-enhancement bias. Prefer manual scoring for those."
    )

    scores: dict = {}
    for model_key, outputs in outputs_by_model.items():
        scores[model_key] = {}
        for concept, text in outputs.items():
            resp = client.chat.completions.create(
                model=judge_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": build_judge_prompt(concept, text)}],
            )
            data = json.loads(resp.choices[0].message.content)
            note = data.get("justification", "")
            if model_key in tested_keys and judge_model.startswith("gpt"):
                note = f"[BIAS: GPT judged GPT] {note}"
            scores[model_key][concept] = {"score": int(data["score"]), "note": note}
    return scores
