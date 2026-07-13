import re
from statistics import pstdev

import textstat


def score_text(text: str) -> dict:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    rows = []
    for sentence in sentences:
        words = len(sentence.split())
        rows.append(
            {
                "words": words,
                "fk": round(textstat.flesch_kincaid_grade(sentence), 2),
            }
        )
    if not rows:
        return {"pass": False, "error": "No sentences generated."}

    dispersion_rows = [row["fk"] for row in rows if row["words"] >= 8]
    if not dispersion_rows:
        dispersion_rows = [row["fk"] for row in rows]
    dispersion = pstdev(dispersion_rows) if len(dispersion_rows) > 1 else 0.0
    whole_fk = round(textstat.flesch_kincaid_grade(text), 2)
    whole_ari = round(textstat.automated_readability_index(text), 2)
    maximum_fk = max(row["fk"] for row in rows)
    long_sentence_pass = all(
        row["words"] < 10 or row["fk"] <= 8.0 for row in rows
    )
    passed = (
        3.0 <= whole_fk <= 6.0
        and 3.0 <= whole_ari <= 7.0
        and dispersion <= 1.7
        and long_sentence_pass
    )
    return {
        "pass": passed,
        "whole_fk": whole_fk,
        "whole_ari": whole_ari,
        "fk_stdev": round(dispersion, 2),
        "max_fk": maximum_fk,
        "sentences": len(rows),
    }
