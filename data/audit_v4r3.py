"""Audit a v4r3 training JSONL before spending on QLoRA."""

import argparse
import json
import sys
from pathlib import Path

from litmus.fk_score import score_text
from data.v4r3 import (
    TARGETED_V4R3_ITEMS,
    V4R3_ARI_BAND,
    V4R3_DISP_MAX,
    V4R3_FK_BAND,
    V4R3_MAX_SENTENCE_FK,
    V4R3_MAX_SENTENCES,
    V4R3_MIN_SENTENCES,
    accuracy_is_2,
    meets_target,
    norm_text,
    target_config,
    training_prompt,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
CONCEPTS_PATH = REPO / "data" / "concepts.json"


def read_jsonl(path: Path) -> list[dict]:
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def passes_accuracy_gate(rec: dict, accuracy_gate: str) -> bool:
    if accuracy_gate == "legacy":
        return accuracy_is_2(rec)
    accuracy = rec.get("accuracy")
    consensus = accuracy.get("consensus") if isinstance(accuracy, dict) else None
    if not isinstance(consensus, dict):
        return False
    if accuracy_gate == "clean-v2":
        return consensus.get("clean_pass") is True
    if accuracy_gate == "tolerant-v2":
        return consensus.get("accuracy_pass_v2") is True
    raise ValueError(f"unknown accuracy gate: {accuracy_gate}")


def audit(path: Path, target: dict, min_sentences: int, max_sentences: int,
          accuracy_gate: str = "legacy", forbid_targeted_v4r3: bool = False) -> dict:
    concepts = json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))
    eval_prompts = {
        norm_text(item)
        for key in ("eval", "calibration_v4r5", "blind_v4r5")
        for item in concepts.get(key, [])
    }
    seen_prompts = {}
    targeted_prompts = {
        norm_text(training_prompt(item)) for item in TARGETED_V4R3_ITEMS
    }
    failures = []
    records = read_jsonl(path)

    for line_no, rec in enumerate(records, 1):
        prompt = training_prompt(rec)
        key = norm_text(prompt)
        concept_key = norm_text(rec["concept"])
        if key in seen_prompts:
            failures.append({
                "line": line_no,
                "kind": "duplicate_prompt",
                "prompt": prompt,
                "first_line": seen_prompts[key],
            })
        else:
            seen_prompts[key] = line_no
        if key in eval_prompts or concept_key in eval_prompts:
            failures.append({"line": line_no, "kind": "eval_leak", "prompt": prompt})
        if not passes_accuracy_gate(rec, accuracy_gate):
            failures.append({
                "line": line_no,
                "kind": f"accuracy_gate_fail:{accuracy_gate}",
                "prompt": prompt,
            })
        if forbid_targeted_v4r3 and key in targeted_prompts:
            failures.append({
                "line": line_no,
                "kind": "historical_targeted_prompt",
                "prompt": prompt,
            })

        score = score_text(rec["explanation"])
        if "error" in score:
            failures.append({"line": line_no, "kind": "score_error", "prompt": prompt})
            continue
        if not score["readability_pass_v4"]:
            failures.append({
                "line": line_no,
                "kind": "v4_readability_fail",
                "prompt": prompt,
                "fk": score["whole_passage_fk"],
                "ari": score["whole_passage_ari"],
                "stdev": score["fk_stdev"],
                "max_fk": score["max_fk"],
            })
        if not meets_target(score, target):
            failures.append({
                "line": line_no,
                "kind": "v4r3_target_fail",
                "prompt": prompt,
                "fk": score["whole_passage_fk"],
                "ari": score["whole_passage_ari"],
                "stdev": score["fk_stdev"],
                "max_fk": score["max_fk"],
            })
        if score["n_sentences"] < min_sentences or score["n_sentences"] > max_sentences:
            failures.append({
                "line": line_no,
                "kind": "sentence_count_fail",
                "prompt": prompt,
                "n_sentences": score["n_sentences"],
            })

    return {
        "path": str(path),
        "records": len(records),
        "unique_prompts": len(seen_prompts),
        "target": target,
        "min_sentences": min_sentences,
        "max_sentences": max_sentences,
        "accuracy_gate": accuracy_gate,
        "forbid_targeted_v4r3": forbid_targeted_v4r3,
        "passed": not failures,
        "failures": failures[:100],
        "failure_count": len(failures),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--fk-min", type=float, default=V4R3_FK_BAND[0])
    ap.add_argument("--fk-max", type=float, default=V4R3_FK_BAND[1])
    ap.add_argument("--ari-min", type=float, default=V4R3_ARI_BAND[0])
    ap.add_argument("--ari-max", type=float, default=V4R3_ARI_BAND[1])
    ap.add_argument("--disp-max", type=float, default=V4R3_DISP_MAX)
    ap.add_argument("--max-sentence-fk", type=float, default=V4R3_MAX_SENTENCE_FK)
    ap.add_argument("--min-sentences", type=int, default=V4R3_MIN_SENTENCES)
    ap.add_argument("--max-sentences", type=int, default=V4R3_MAX_SENTENCES)
    ap.add_argument(
        "--accuracy-gate",
        choices=("legacy", "clean-v2", "tolerant-v2"),
        default="legacy",
    )
    ap.add_argument("--forbid-targeted-v4r3", action="store_true")
    args = ap.parse_args()

    target = target_config(
        fk_min=args.fk_min,
        fk_max=args.fk_max,
        ari_min=args.ari_min,
        ari_max=args.ari_max,
        disp_max=args.disp_max,
        max_sentence_fk=args.max_sentence_fk,
    )
    summary = audit(
        Path(args.path),
        target,
        args.min_sentences,
        args.max_sentences,
        args.accuracy_gate,
        args.forbid_targeted_v4r3,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
