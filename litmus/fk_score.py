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
WP_BAND = (1.5, 3.0)    # floor 1.5: textstat rates clear grade-3 prose ~1.5-2.0,
                        # so a 2.0 floor forced over-enrichment; 1.5 still rejects
                        # baby-talk. Ceiling 3.0 keeps it readable.
DISPERSION_MAX = 1.3    # allow natural variation; spikes still caught by backstop
DISP_MIN_WORDS = 8
LONG_MIN_WORDS = 10
BACKSTOP_CEILING = 4.0

# --- v4 gate (Day-4 recalibration) -------------------------------------------
# The v3 band (whole-passage FK 1.5-3.0) was calibrated against AI-authored text
# that had itself been iterated on FK, so it was circular. Re-scored against the
# CommonLit CLEAR corpus (grade labels from teacher pairwise judgments, no formula
# involved -- see eval/metric_comparison_real.md), GENUINE grade-3 informational
# prose averages whole-passage FK ~5.5 (IQR ~3.5-8), with ARI right beside it
# (~5.9). Only 5.9% of real grade-3 text passes FK<=3.0 -- so the v3 band was
# actually targeting ~grade 1-2 vocabulary, not grade 3.
# On those independent labels FK and ARI were the two best separators (AUC 0.990
# / 0.988), statistically tied and ahead of SMOG/Coleman-Liau/Dale-Chall. So v4:
#   1. whole-passage FK in WP_BAND_V4 (3.0-6.0): real grade-3 reading level, still
#      short-sentence (the floor rejects baby-talk, the ceiling keeps it grade-3).
#   2. whole-passage ARI in ARI_BAND_V4 (3.0-7.0): the co-best metric as a second
#      gate. ARI is character-based, so it fails differently from FK (syllable-
#      based) and catches passages one of them misjudges.
#   3. dispersion cap (DISPERSION_MAX_V4) + long-sentence backstop, same idea as
#      v3 but scaled to the higher band.
WP_BAND_V4 = (3.0, 6.0)
ARI_BAND_V4 = (3.0, 7.0)
DISPERSION_MAX_V4 = 1.7    # looser than v3's 1.3: the wider grade-3 band tolerates more
                           # sentence-to-sentence variation, but still rejects passages
                           # that lurch between baby-talk and hard sentences. Egregious
                           # run-ons are also caught by the backstop below.
BACKSTOP_CEILING_V4 = 8.0  # no >=10-word sentence may blow past the band ceiling


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

    # v4 gate (Day-4 recalibration; FK + ARI, band 3-6). See constants block.
    whole_passage_ari = round(textstat.automated_readability_index(text), 2)
    long_over_v4 = [r for r in rows
                    if r["words"] >= LONG_MIN_WORDS and r["fk"] > BACKSTOP_CEILING_V4]
    cond_wp_band_v4 = WP_BAND_V4[0] <= whole_passage_fk <= WP_BAND_V4[1]
    cond_ari_band_v4 = ARI_BAND_V4[0] <= whole_passage_ari <= ARI_BAND_V4[1]
    cond_dispersion_v4 = fk_stdev <= DISPERSION_MAX_V4
    cond_backstop_v4 = len(long_over_v4) == 0
    readability_pass_v4 = (cond_wp_band_v4 and cond_ari_band_v4
                           and cond_dispersion_v4 and cond_backstop_v4)

    return {
        "n_sentences": len(rows),
        "max_fk": max(r["fk"] for r in rows),
        "n_over_ceiling": len(over),
        "n_long_over_ceiling": len(long_over),  # >= LONG_MIN_WORDS sents over BACKSTOP_CEILING
        "n_short_flag": sum(1 for r in rows if r["short_flag"]),
        "pct_in_band": round(pct_in_band, 2),        # DIAGNOSTIC ONLY (original 2.0-3.0 band)
        "whole_passage_fk": whole_passage_fk,
        "whole_passage_ari": whole_passage_ari,      # v4 co-metric
        "fk_stdev": round(fk_stdev, 2),              # dispersion (over >= DISP_MIN_WORDS sents)
        "readability_pass": readability_pass,        # original litmus gate (per-sentence)
        "readability_pass_v3": readability_pass_v3,  # Day-3 gate (whole-passage FK 1.5-3.0)
        "readability_pass_v4": readability_pass_v4,  # OPERATIVE gate (Day-4: FK 3-6 AND ARI 3-7)
        "cond_wp_band": cond_wp_band,
        "cond_dispersion": cond_dispersion,
        "cond_backstop": cond_backstop,
        "cond_wp_band_v4": cond_wp_band_v4,
        "cond_ari_band_v4": cond_ari_band_v4,
        "cond_dispersion_v4": cond_dispersion_v4,
        "cond_backstop_v4": cond_backstop_v4,
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
