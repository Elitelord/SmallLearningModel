"""Flesch-Kincaid readability scoring — the reusable ruler for the whole week.

Per-sentence scoring is the whole point: the Behavior Spec forbids ANY sentence
over FK 3.0, so we score sentences individually, not just the whole passage.
Very short sentences get an FK score that is unreliable (the formula is
length-driven), so we FLAG them rather than silently trusting them.

`readability_pass` is only HALF the spec. Accuracy is scored separately in
accuracy.py, and overall_pass = readability_pass AND accuracy_pass.
"""

import re
from statistics import pstdev

import textstat

CEILING = 3.0           # no sentence may exceed this FK grade (original litmus gate)
BAND = (2.0, 3.0)       # original litmus target band
BAND_MIN_PASS = 0.70    # >=70% of sentences must fall inside the band
MIN_WORDS = 4           # sentences under this are FK-unreliable -> flagged

# --- v3 gate (operative; Day-3 spec decision) --------------------------------
# The original per-sentence band (2.0-3.0 + hard ceiling 3.0) proved unusable as a
# data-gen gate: textstat's FK is noise on short sentences (an 8-word plain
# sentence can score FK 8), so <=5% of genuinely-simple grade-3 explanations pass.
# The v3 gate encodes the real intent -- "grade 3, on average, evenly, no spikes":
#   1. whole-passage FK in WP_BAND (2.0-3.0): floor + ceiling. The floor rejects
#      trivially-easy baby-talk; the ceiling keeps it readable. Whole-passage FK
#      is length-dampened, so it is robust to short-sentence noise.
#   2. dispersion cap: std-dev of per-sentence FK (over sentences >= DISP_MIN_WORDS
#      words, to skip noisy short ones) <= DISPERSION_MAX. This encodes "evenly" --
#      no lurching between baby-talk and hard sentences. Calibrated on 37 Claude
#      drafts: even grade-3 passages have std-dev <= ~0.87, spiky ones ~1.5+.
#   3. backstop: no sentence with >= LONG_MIN_WORDS words exceeds BACKSTOP_CEILING
#      (guards against a single wildly-hard long sentence a low std-dev could hide).
# The old >=70%-in-band rule is DROPPED; pct_in_band is still reported as a
# diagnostic. Re-scoring the litmus frontier outputs under v3 still shows them
# failing, so the "fine-tuning is justified" baseline conclusion is preserved.
WP_BAND = (2.0, 3.0)
DISPERSION_MAX = 1.0
DISP_MIN_WORDS = 8
LONG_MIN_WORDS = 10
BACKSTOP_CEILING = 4.0


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

    # v3 gate (operative for data-gen + eval). See the constants block for rationale.
    whole_passage_fk = round(textstat.flesch_kincaid_grade(text), 2)
    disp_rows = [r["fk"] for r in rows if r["words"] >= DISP_MIN_WORDS] or [r["fk"] for r in rows]
    fk_stdev = pstdev(disp_rows) if len(disp_rows) > 1 else 0.0
    long_over = [r for r in rows
                 if r["words"] >= LONG_MIN_WORDS and r["fk"] > BACKSTOP_CEILING]
    cond_wp_band = WP_BAND[0] <= whole_passage_fk <= WP_BAND[1]
    cond_dispersion = fk_stdev <= DISPERSION_MAX
    cond_backstop = len(long_over) == 0
    readability_pass_v3 = cond_wp_band and cond_dispersion and cond_backstop

    return {
        "n_sentences": len(rows),
        "max_fk": max(r["fk"] for r in rows),
        "n_over_ceiling": len(over),
        "n_long_over_ceiling": len(long_over),  # >= LONG_MIN_WORDS sents over BACKSTOP_CEILING
        "n_short_flag": sum(1 for r in rows if r["short_flag"]),
        "pct_in_band": round(pct_in_band, 2),        # DIAGNOSTIC ONLY (original 2.0-3.0 band)
        "whole_passage_fk": whole_passage_fk,
        "fk_stdev": round(fk_stdev, 2),              # dispersion (over >= DISP_MIN_WORDS sents)
        "readability_pass": readability_pass,        # original litmus gate (per-sentence)
        "readability_pass_v3": readability_pass_v3,  # OPERATIVE gate (Day-3 decision)
        "cond_wp_band": cond_wp_band,
        "cond_dispersion": cond_dispersion,
        "cond_backstop": cond_backstop,
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
