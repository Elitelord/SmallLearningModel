"""Add accuracy scores to an ALREADY-GENERATED base_vs_tuned results JSON.

The base_vs_tuned run stores base_text/tuned_text for every concept, so accuracy
can be judged AFTER the fact from any machine with gateway creds — no GPU, no model
reload. This is the path the Colab notebook documents for the accuracy delta.

For each row it judges base_text and tuned_text with the 0/1/2 mechanism rubric
(audience-calibrated, judge != student), fills in `accuracy` + recomputes
`overall_pass = readability_pass AND accuracy == 2`, and writes a *_judged.json.

    .venv\\Scripts\\python -m eval.judge_saved base_vs_tuned_v4r3_litmus12_readability.json \\
        --judge openai-group/gpt-4.1 --concurrency 8
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from litmus.accuracy import build_judge_prompt
from litmus.env import make_client

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def judge_one(client, judge_model, concept, text, tries=5):
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user",
                           "content": build_judge_prompt(concept, text, audience_calibrated=True)}],
            )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                raise RuntimeError("empty judge content")
            data = json.loads(content)
            return int(data["score"]), data.get("justification", "")
        except Exception as e:  # noqa: BLE001
            if attempt >= tries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def judge_file(path: Path, judge_model: str, concurrency: int) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]

    # one task per (row, side) that has text and no accuracy yet
    tasks = []
    for i, r in enumerate(rows):
        for side, tkey in (("base", "base_text"), ("tuned", "tuned_text")):
            text = r.get(tkey)
            if text and r[side].get("accuracy") is None:
                tasks.append((i, side, r["concept"], text))

    client = make_client()
    print(f"[{path.name}] judging {len(tasks)} outputs with {judge_model} "
          f"(concurrency={concurrency})", flush=True)

    def work(task):
        i, side, concept, text = task
        acc, note = judge_one(client, judge_model, concept, text)
        return i, side, acc, note

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            i, side, acc, note = fut.result()
            r = rows[i]
            r[side]["accuracy"] = acc
            r[side]["note"] = note
            r[side]["overall_pass"] = bool(r[side]["readability_pass"] and acc == 2)
            done += 1
            print(f"  [{done}/{len(tasks)}] {side:5} acc={acc} | {r['concept'][:50]}", flush=True)

    data["judge"] = judge_model
    out_path = path.with_name(path.name.replace("_readability", "").replace(".json", "") + "_judged.json")
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    n = len(rows)
    def cnt(side, pred): return sum(1 for r in rows if pred(r[side]))
    print(f"\n[{path.name}] n={n}")
    for side in ("base", "tuned"):
        rp = cnt(side, lambda s: s["readability_pass"])
        ac = cnt(side, lambda s: s.get("accuracy") == 2)
        op = cnt(side, lambda s: s.get("overall_pass"))
        print(f"  {side:5}: readability {rp}/{n}  accuracy=2 {ac}/{n}  overall {op}/{n}")
    print(f"  wrote {out_path}\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="base_vs_tuned results JSON(s) to judge")
    ap.add_argument("--judge", default="openai-group/gpt-4.1")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    for p in args.paths:
        judge_file(Path(p), args.judge, args.concurrency)


if __name__ == "__main__":
    main()
