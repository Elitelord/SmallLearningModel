# data/ — generation pipeline (Part A)

The dataset is the real deliverable. This directory builds it with a
**generate → readability-rewrite → accuracy-gate** pipeline that reuses the
litmus harness as both the readability filter and the accuracy judge.

## Files

| File | What it is |
|---|---|
| `sft_format.py` | The **minimal** SFT prompt (`Explain: {concept}`), the single source of truth for train + inference so they never drift. Part B. |
| `gen_concepts.py` | Part A.1 — generate the concept list via the teacher, dedup, exclude the 12 litmus concepts, write `concepts.json` with a train/eval split. |
| `exemplars.json` | Part A.2 — 4 hand-authored, human-verified grade-3 gold explanations, each **passing** `litmus.fk_score.score_text` and conveying the mechanism. Few-shot anchors. Concepts here are NOT among the 12 litmus concepts. |
| `generate.py` | Part A.3–A.5 — the generate/rewrite/accuracy-gate loop with yield logging. |
| `concepts.json` | Output of `gen_concepts.py`: `{eval: [12 litmus concepts], train: [...]}`. |
| `generated_v0.jsonl` | Output of `generate.py`: one kept example per line. (v0 = smoke-test junk.) |
| `generated_v0.stats.json` | Yield report (discards by stage, avg rewrite iters). |

## Guardrails baked in

- **Narrow domain**: elementary physical + life science only. A 0.6B student learns *form* faster than *facts*; a wide domain yields fluent-but-wrong grade-3 text.
- **Strict accuracy gate**: keep only `accuracy == 2` (correct AND real mechanism). 1s and 0s are discarded — a wrong example is poison for a small student.
- **No leakage**: the 12 litmus concepts are the eval set and are removed from train by normalized-string match.
- **Judge ≠ student, judge ≠ teacher (family caveat)**: student = Qwen; teacher = `gpt-4o-mini`; judge = `gpt-4o`. Judge is a different, stronger model than the teacher. The only API key here is OpenAI, so judge and teacher share a family — a residual self-enhancement risk (per MT-Bench) noted as a limitation. Day 3 can revisit (e.g. a non-OpenAI judge).

## Run

```
# Part A.1 — concepts (real run ~300; smoke ~60; --offline uses a seed list)
.venv\Scripts\python -m data.gen_concepts --target 300
# Part A.3–5 — full pipeline (rewrite loop + accuracy gate)
.venv\Scripts\python -m data.generate
# SMOKE (Part E) — 50 quick junk examples, no filtering
.venv\Scripts\python -m data.generate --junk --limit 50 --out data/generated_v0.jsonl
```

## Day-2 finding (viability signal for Day 3)

Running the **full** pipeline with `teacher=gpt-4o-mini` discarded 3/3 sampled
concepts on the **readability** gate (0% yield): the model gets under the FK 3.0
ceiling but can't reliably keep ≥70% of sentences inside the narrow 2.0–3.0 band
(too many trivially short sentences fall below 2.0). This matches the litmus
conclusion that prompting alone can't hit the spec reliably — and tells us Day 3
should likely use a stronger teacher (`gpt-4o`) and/or accept a wider band before
generating the real v1 dataset. The `--junk` smoke path bypasses the gates, so
Part E is unaffected.
