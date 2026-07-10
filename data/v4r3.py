"""Shared v4r3 training-data target and audit helpers."""

from __future__ import annotations

import re
from copy import deepcopy

V4R3_TARGET_NAME = "v4r3"
V4R3_FK_BAND = (3.3, 5.0)
V4R3_ARI_BAND = (3.8, 6.2)
V4R3_DISP_MAX = 1.1
V4R3_MAX_SENTENCE_FK = 7.0
V4R3_MIN_SENTENCES = 4
V4R3_MAX_SENTENCES = 6

TARGETED_V4R3_ITEMS = [
    {
        "concept": "How does sunlight scatter in air?",
        "phrasing": "How does sunlight scatter in air?",
    },
    {
        "concept": "Why can water in a puddle go into the air?",
        "phrasing": "Why can water in a puddle go into the air?",
    },
    {
        "concept": "How does Earth's tilt change sunlight?",
        "phrasing": "How does Earth's tilt change sunlight?",
    },
    {
        "concept": "How do lungs trade oxygen and carbon dioxide?",
        "phrasing": "How do lungs trade oxygen and carbon dioxide?",
    },
    {
        "concept": "How do fish gills take oxygen from water?",
        "phrasing": "How do fish gills take oxygen from water?",
    },
    {
        "concept": "What makes magnets pull on iron?",
        "phrasing": "What makes magnets pull on iron?",
    },
    {
        "concept": "Why does the Moon show different phases?",
        "phrasing": "Why does the Moon show different phases?",
    },
    {
        "concept": "How do raindrops split sunlight into colors?",
        "phrasing": "How do raindrops split sunlight into colors?",
    },
]


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def training_prompt(rec: dict) -> str:
    return rec.get("phrasing") or rec["concept"]


def target_config(
    *,
    fk_min: float = V4R3_FK_BAND[0],
    fk_max: float = V4R3_FK_BAND[1],
    ari_min: float = V4R3_ARI_BAND[0],
    ari_max: float = V4R3_ARI_BAND[1],
    disp_max: float = V4R3_DISP_MAX,
    max_sentence_fk: float = V4R3_MAX_SENTENCE_FK,
    name: str = V4R3_TARGET_NAME,
) -> dict:
    return {
        "name": name,
        "fk_band": [fk_min, fk_max],
        "ari_band": [ari_min, ari_max],
        "disp_max": disp_max,
        "max_sentence_fk": max_sentence_fk,
    }


def meets_target(score: dict, target: dict | None = None) -> bool:
    if target is None:
        target = target_config()
    if "error" in score:
        return False
    fk_lo, fk_hi = target["fk_band"]
    ari_lo, ari_hi = target["ari_band"]
    return (
        fk_lo <= score["whole_passage_fk"] <= fk_hi
        and ari_lo <= score["whole_passage_ari"] <= ari_hi
        and score["fk_stdev"] <= target["disp_max"]
        and score["max_fk"] <= target["max_sentence_fk"]
        and score["readability_pass_v4"]
    )


def accuracy_is_2(rec: dict) -> bool:
    accuracy = rec.get("accuracy")
    if isinstance(accuracy, dict):
        return accuracy.get("score") == 2
    return accuracy == 2


def with_current_fk(rec: dict, score: dict, target: dict) -> dict:
    out = deepcopy(rec)
    out["fk"] = {
        "max_fk": score["max_fk"],
        "whole_passage_fk": score["whole_passage_fk"],
        "whole_passage_ari": score["whole_passage_ari"],
        "fk_stdev": score["fk_stdev"],
        "readability_pass_v4": score["readability_pass_v4"],
    }
    out["n_sentences"] = score["n_sentences"]
    out["generation_target"] = target
    return out
