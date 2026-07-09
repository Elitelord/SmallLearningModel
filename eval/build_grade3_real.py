"""Build an INDEPENDENT grade-3 ground-truth test set from the CommonLit CLEAR corpus.

Why CLEAR: its difficulty labels are formula-INDEPENDENT — a `Lexile Band` per
excerpt plus `BT_easiness`, a Bradley-Terry ease score derived from teacher
pairwise judgments (NOT from any readability formula). That is exactly what the
FK-vs-other-metrics study was missing: the old eval set (data/sample/
fk_eval_drafts_37.jsonl) used AI-authored positives that had been ITERATED against
FK, so FK was structurally advantaged. Here no metric saw the labels.

Lexile -> grade (MetaMetrics grade bands, approx):
    band 500  ~ grade 2-3      band 900  ~ grade 4-5      band 1300 ~ grade 9-11
    band 700  ~ grade 3-4      band 1100 ~ grade 6-8      band 1500 ~ grade 11-12+

Test-set design (Info category only, to stay near the science-explanation domain):
    POSITIVE (label_good=True)  = real grade-3 band  : Lexile band 500 or 700
    NEGATIVE (label_good=False) = adult band         : Lexile band 1300 or 1500
Positives are kept in full (n~150). Negatives are stratified-sampled to match.

We DO NOT trust CLEAR's precomputed formula columns (different textstat build);
run_grade3_real.py recomputes every metric with the repo's textstat 0.7.13.

Also emits a per-band gradient (300..1500) so we can see WHERE each metric places
genuine grade-3 material relative to the project's FK 1.5-3.0 gate.

Usage:
    .venv\\Scripts\\python.exe eval/build_grade3_real.py            # fetch + build
    .venv\\Scripts\\python.exe eval/build_grade3_real.py --xlsx path/to/clear.xlsx
"""

import argparse
import io
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "sample" / "grade3_real_clear.jsonl"
GRADIENT_OUT = REPO / "eval" / "_grade3_real_gradient.json"
CLEAR_URL = ("https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/"
             "main/CLEAR_corpus_final.xlsx")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

POS_BANDS = {"500", "700", "410L-600L", "610L-800L",
             "410L - 600L", "610L - 800L", "610L-800"}
NEG_BANDS = {"1300", "1500"}
GRADIENT_BANDS = ["300", "500", "700", "900", "1100", "1300", "1500", "1700"]


def _colidx(ref: str) -> int:
    import re
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def parse_xlsx(raw: bytes) -> list[dict]:
    z = zipfile.ZipFile(io.BytesIO(raw))
    sroot = ET.fromstring(z.read("xl/sharedStrings.xml"))
    sst = ["".join(t.text or "" for t in si.iter(NS + "t"))
           for si in sroot.findall(NS + "si")]

    def cv(c):
        t = c.get("t")
        v = c.find(NS + "v")
        if v is None:
            istr = c.find(NS + "is")
            return "".join(x.text or "" for x in istr.iter()) if istr is not None else ""
        return sst[int(v.text)] if t == "s" else v.text

    root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in root.find(NS + "sheetData").findall(NS + "row"):
        cells = {_colidx(c.get("r")): cv(c) for c in r.findall(NS + "c")}
        rows.append([cells.get(i, "") for i in range(28)])
    hdr = rows[0]
    return [dict(zip(hdr, r)) for r in rows[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=None, help="local CLEAR xlsx (else fetch)")
    ap.add_argument("--neg-per-pos", type=float, default=1.0,
                    help="how many negatives per positive (1.0 = balanced)")
    args = ap.parse_args()

    if args.xlsx:
        raw = Path(args.xlsx).read_bytes()
        print(f"[local] {args.xlsx} ({len(raw)} bytes)")
    else:
        print(f"[fetch] {CLEAR_URL}")
        req = urllib.request.Request(CLEAR_URL, headers={"User-Agent": "research/1.0"})
        raw = urllib.request.urlopen(req, timeout=90).read()
        print(f"  downloaded {len(raw)} bytes")

    recs = parse_xlsx(raw)
    info = [r for r in recs if r["Categ"] == "Info"]
    print(f"CLEAR total={len(recs)}  Info={len(info)}")

    def band(r):
        return r["Lexile Band"].strip()

    pos = [r for r in info if band(r) in POS_BANDS]
    neg_all = [r for r in info if band(r) in NEG_BANDS]
    # stratified, deterministic negative sample: sort by ID, stride to target count.
    target_neg = int(round(len(pos) * args.neg_per_pos))
    neg_all.sort(key=lambda r: str(r["ID"]))
    if target_neg < len(neg_all):
        stride = len(neg_all) / target_neg
        neg = [neg_all[int(i * stride)] for i in range(target_neg)]
    else:
        neg = neg_all
    print(f"positives (band 500/700 Info) = {len(pos)}")
    print(f"negatives (band 1300/1500 Info) = {len(neg)}  (of {len(neg_all)} available)")

    def to_row(r, good):
        return {
            "text": " ".join(r["Excerpt"].split()),
            "label_good": good,          # True = real grade-3 material
            "source": "CLEAR",
            "lexile_band": band(r),
            "bt_easiness": r.get("BT_easiness", ""),
            "categ": r["Categ"],
            "sub_cat": r.get("Sub Cat", ""),
            "license": r.get("License", "") or "(unspecified)",
            "clear_id": r["ID"],
            "title": r.get("Title", ""),
        }

    rows = [to_row(r, True) for r in pos] + [to_row(r, False) for r in neg]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT}  ({len(rows)} rows: {len(pos)} pos / {len(neg)} neg)")

    # per-band gradient rows (all Info), for the recalibration view (scored later)
    gradient = {}
    for b in GRADIENT_BANDS:
        band_rows = [to_row(r, band(r) in POS_BANDS) for r in info if band(r) == b]
        gradient[b] = band_rows
    GRADIENT_OUT.write_text(json.dumps(gradient, ensure_ascii=False), encoding="utf-8")
    counts = {b: len(v) for b, v in gradient.items()}
    print(f"Wrote {GRADIENT_OUT}  (per-band Info counts: {counts})")


if __name__ == "__main__":
    main()
