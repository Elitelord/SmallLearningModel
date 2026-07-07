"""Ergonomic paste front-end for the browser models.

Pasting prose straight into manual_outputs.json means hand-escaping quotes and
newlines. Instead: paste each answer into a plain .md file under simple markers,
then run this to convert -> valid JSON (all escaping handled).

Step 1 — make blank templates (one per browser model):
    .venv\\Scripts\\python -m litmus.paste --init

    Creates litmus/paste/gemini.md and litmus/paste/claude_browser.md, each with
    one "@@@ <concept>" marker per concept and a blank line to paste under.

Step 2 — paste each model's answer under its marker (raw prose is fine).

Step 3 — ingest into manual_outputs.json:
    .venv\\Scripts\\python -m litmus.paste

    Reads every litmus/paste/*.md, fills manual_outputs.json (skips blanks),
    and reports how many concepts each model has. Then run:  python -m litmus.score_all
"""

import argparse
import json
import sys
from pathlib import Path

from litmus.concepts import CONCEPTS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PASTE_DIR = HERE / "paste"
MANUAL_PATH = HERE / "manual_outputs.json"
MARKER = "@@@ "
BROWSER_MODELS = ["gemini", "claude_browser"]


def init_templates():
    PASTE_DIR.mkdir(exist_ok=True)
    for model in BROWSER_MODELS:
        lines = [
            f"# Paste {model} answers below each marker. Raw prose is fine — do NOT",
            "# add quotes or JSON. One answer per marker. Leave blanks to skip.",
            "",
        ]
        for c in CONCEPTS:
            lines.append(f"{MARKER}{c}")
            lines.append("")
            lines.append("")
        path = PASTE_DIR / f"{model}.md"
        if path.exists():
            print(f"exists, not overwriting: {path}")
            continue
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {path}")
    print("\nPaste answers under each '@@@' marker, then run: python -m litmus.paste")


def parse_file(path: Path) -> dict:
    """Parse an @@@-delimited paste file into {concept: text}."""
    text = path.read_text(encoding="utf-8")
    valid = {c.lower(): c for c in CONCEPTS}
    out, current, buf = {}, None, []

    def flush():
        if current is not None:
            body = "\n".join(buf).strip()
            if body:
                out[current] = body

    for line in text.splitlines():
        if line.startswith("#") and current is None:
            continue  # header comment before first marker
        if line.startswith(MARKER):
            flush()
            header = line[len(MARKER):].strip()
            canon = valid.get(header.lower())
            if canon is None:
                print(f"  WARNING: unknown concept header in {path.name}: {header!r}")
                current = None
            else:
                current = canon
            buf = []
        else:
            buf.append(line)
    flush()
    return out


def ingest():
    if not PASTE_DIR.exists():
        print("No litmus/paste/ dir. Run:  python -m litmus.paste --init")
        return
    manual = json.loads(MANUAL_PATH.read_text(encoding="utf-8")) if MANUAL_PATH.exists() else {}

    for path in sorted(PASTE_DIR.glob("*.md")):
        model = path.stem
        parsed = parse_file(path)
        if not parsed:
            continue
        manual.setdefault(model, {c: "" for c in CONCEPTS})
        manual[model].update(parsed)
        missing = [c for c in CONCEPTS if not manual[model].get(c, "").strip()]
        print(f"{model}: {len(CONCEPTS) - len(missing)}/{len(CONCEPTS)} concepts filled"
              + (f"  (still missing: {len(missing)})" if missing else ""))

    MANUAL_PATH.write_text(json.dumps(manual, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {MANUAL_PATH}\nNext: score accuracy in accuracy_scores.json, then  python -m litmus.score_all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="create blank paste templates")
    args = ap.parse_args()
    if args.init:
        init_templates()
    else:
        ingest()


if __name__ == "__main__":
    main()
