"""Flesch-Kincaid readability scoring — the reusable ruler for the whole week.

Per-sentence scoring is the whole point: the Behavior Spec forbids ANY sentence
over FK 3.0, so we score sentences individually, not just the whole passage.
Very short sentences get an FK score that is unreliable (the formula is
length-driven), so we FLAG them rather than silently trusting them.

`readability_pass` is only HALF the spec. Accuracy is scored separately in
accuracy.py, and overall_pass = readability_pass AND accuracy_pass.
"""

import re

import textstat

CEILING = 3.0           # no sentence may exceed this FK grade (original litmus gate)
BAND = (2.0, 3.0)       # original litmus target band
BAND_MIN_PASS = 0.70    # >=70% of sentences must fall inside the band
MIN_WORDS = 4           # sentences under this are FK-unreliable -> flagged

# --- v2 gate (Day-3 spec decision) -------------------------------------------
# The original per-sentence band (2.0-3.0 + hard ceiling 3.0) proved unusable as a
# data-gen gate: textstat's FK is noise on short sentences (an 8-word plain
# sentence can score FK 8), so <=5% of genuinely-simple grade-3 explanations pass,
# and neither hand-authoring nor best-of-N model sampling can hit it. The v2 gate
# keeps the "a 3rd grader can read it" intent while dropping the metric noise:
#   1. whole-passage FK <= CEILING            (robust, length-dampened)
#   2. no sentence with >= LONG_MIN_WORDS words exceeds V2_LONG_CEILING
#      (short sentences exempt; long ones get a lenient 4.0 bound, not 3.0,
#       since even 10-word sentences carry FK noise)
#   3. >= BAND_MIN_PASS of ALL sentences fall in V2_BAND (1.0-3.0)
# Re-scoring the litmus frontier outputs under v2 still shows them failing
# (gpt 1/12, gemini 0/12, claude 1/12, qwen 0/12), so the "fine-tuning is
# justified" baseline conclusion is preserved.
LONG_MIN_WORDS = 10
V2_BAND = (1.0, 3.0)
V2_LONG_CEILING = 4.0


def split_sentences(text: str):
    """Split into sentences.

    Uses a punctuation regex by default (matches the reference impl and needs no
    downloaded corpora, so results are reproducible). If nltk punkt data happens
    to be installed locally it is preferred, since it handles abbreviations and
    decimals better; we never trigger a network download.
    """
    text = text.strip()
    if not text:
        return []
    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        # Only use nltk if the tokenizer data is already present locally.
        nltk.data.find("tokenizers/punkt_tab")
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in parts if s.strip()]


def score_text(text: str) -> dict:
    """Score one passage. Returns per-sentence rows + passage-level rollup.

    `readability_pass` requires BOTH: zero sentences over the ceiling AND at
    least BAND_MIN_PASS of sentences inside the band.
    """
    sents = split_sentences(text)
    rows = []
    for s in sents:
        wc = len(s.split())
        fk = textstat.flesch_kincaid_grade(s)
        rows.append(
            {
                "sentence": s,
                "words": wc,
                "fk": round(fk, 2),
                "over_ceiling": fk > CEILING,
                "short_flag": wc < MIN_WORDS,  # FK unreliable on very short sentences
            }
        )
    if not rows:
        return {"error": "no sentences"}

    in_band = [r for r in rows if BAND[0] <= r["fk"] <= BAND[1]]
    over = [r for r in rows if r["over_ceiling"]]
    pct_in_band = len(in_band) / len(rows)
    readability_pass = (len(over) == 0) and (pct_in_band >= BAND_MIN_PASS)

    # v2 gate (operative for data-gen + eval, per the Day-3 spec decision). See the
    # constants block above for the rationale. Band is V2_BAND (1.0-3.0), the long
    # ceiling is V2_LONG_CEILING (4.0) enforced only on >= LONG_MIN_WORDS sentences.
    whole_passage_fk = round(textstat.flesch_kincaid_grade(text), 2)
    pct_in_v2_band = sum(1 for r in rows if V2_BAND[0] <= r["fk"] <= V2_BAND[1]) / len(rows)
    long_over = [r for r in rows
                 if r["words"] >= LONG_MIN_WORDS and r["fk"] > V2_LONG_CEILING]
    cond_whole_passage = whole_passage_fk <= CEILING
    cond_long_ceiling = len(long_over) == 0
    cond_band = pct_in_v2_band >= BAND_MIN_PASS
    readability_pass_v2 = cond_whole_passage and cond_long_ceiling and cond_band

    return {
        "n_sentences": len(rows),
        "max_fk": max(r["fk"] for r in rows),
        "n_over_ceiling": len(over),
        "n_long_over_ceiling": len(long_over),  # >= LONG_MIN_WORDS sents over V2_LONG_CEILING
        "n_short_flag": sum(1 for r in rows if r["short_flag"]),
        "pct_in_band": round(pct_in_band, 2),        # original 2.0-3.0 band
        "pct_in_v2_band": round(pct_in_v2_band, 2),  # v2 1.0-3.0 band
        "whole_passage_fk": whole_passage_fk,
        "readability_pass": readability_pass,  # original litmus gate (per-sentence)
        "readability_pass_v2": readability_pass_v2,  # operative gate (Day-3 decision)
        "cond_whole_passage": cond_whole_passage,
        "cond_long_ceiling": cond_long_ceiling,
        "cond_band": cond_band,
        "sentences": rows,
    }


if __name__ == "__main__":
    demo = (
        "The sky is blue because of sunlight. Sunlight has many colors mixed "
        "in it. Air scatters the blue light the most. So we see blue when we "
        "look up."
    )
    import json

    print(json.dumps(score_text(demo), indent=2))
