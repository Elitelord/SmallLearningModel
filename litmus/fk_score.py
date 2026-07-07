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

CEILING = 3.0           # no sentence may exceed this FK grade
BAND = (2.0, 3.0)       # the target band
BAND_MIN_PASS = 0.70    # >=70% of sentences must fall inside the band
MIN_WORDS = 4           # sentences under this are FK-unreliable -> flagged


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
    return {
        "n_sentences": len(rows),
        "max_fk": max(r["fk"] for r in rows),
        "n_over_ceiling": len(over),
        "n_short_flag": sum(1 for r in rows if r["short_flag"]),
        "pct_in_band": round(pct_in_band, 2),
        "whole_passage_fk": round(textstat.flesch_kincaid_grade(text), 2),
        "readability_pass": readability_pass,  # HALF the spec — accuracy is separate
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
