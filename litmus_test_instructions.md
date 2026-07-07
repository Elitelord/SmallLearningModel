# Litmus Test + Eval Harness — Instructions for Claude Code

## Objective

Test the assignment's litmus question — **"can a well-prompted model already do this reliably?"** — against the Behavior Spec, across four models. The output is (a) a pass-rate table that tells us whether fine-tuning is justified, and (b) a reusable FK-scoring eval harness we'll use all week.

**Behavior Spec (the thing being tested):**
> Given any elementary physical- or life-science concept, produce an explanation in which **no sentence exceeds Flesch-Kincaid grade 3.0**, **≥70% of sentences fall within FK 2.0–3.0**, and the explanation **stays factually correct and conveys the core mechanism** (not just a definition).
>
> Forbidden failure: any single sentence above FK 3.0, OR any factual error / oversimplification that misrepresents the mechanism.

The key word is **reliably** — we care about pass rate across many concepts, not one lucky output.

## Models under test

| Model | How to run |
|---|---|
| GPT (e.g. gpt-4o / gpt-4.1) | **API** — automate in the harness (key in env) |
| Gemini (browser) | Manual paste — no API key |
| Claude (browser, separate session) | Manual paste |
| Qwen3 0.6B or 1.7B **Instruct** (the tune target) | Run via transformers/Unsloth in this repo if it loads, else generate in Colab and paste |

Use the **same prompt and same concept list** for every model. Note the temperature you use for API/Qwen (suggest ~0.7). Frontier browser temps aren't controllable — record that as a limitation.

## Fixed concept set (reuse ALL WEEK — same ruler as Day-3 eval)

1. Why is the sky blue?
2. How do plants make their own food?
3. Why do we have day and night?
4. What makes ice melt?
5. How do magnets work?
6. Why do things fall to the ground?
7. Where does a puddle go when it dries up?
8. Why do we have seasons?
9. How do our lungs help us breathe?
10. What makes a rainbow?
11. Why does the moon look like it changes shape?
12. How do fish breathe underwater?

## The eval prompt (paste verbatim into each browser model; use as the user message for API/Qwen)

```
You are explaining science to a 7-year-old in the 3rd grade.

Rules:
- Every single sentence must be simple enough for an 8-year-old to read on their own. Use short sentences and common, everyday words.
- Avoid technical terms. If you must use one, explain it in plain words right away.
- Stay scientifically accurate. Explain HOW or WHY it really works — the actual reason — not just what it is.
- Do NOT oversimplify so much that it becomes wrong.
- Write 4 to 8 sentences.

Explain: {CONCEPT}
```

(Optional harder variant: prepend 1–2 worked examples to make it a few-shot prompt. If even few-shot drifts, that's an even stronger result. Test the zero-shot version first.)

## FK scoring — reference implementation

Per-sentence scoring is the whole point (the spec forbids ANY sentence over 3.0), so score sentences individually, not the whole passage. Handle the short-sentence artifact: FK on a 3-word sentence is unreliable, so flag (don't silently trust) very short sentences.

```python
# pip install textstat
import textstat, re

CEILING = 3.0
BAND = (2.0, 3.0)
BAND_MIN_PASS = 0.70
MIN_WORDS = 4  # sentences under this get flagged as FK-unreliable

def split_sentences(text: str):
    text = text.strip()
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]
    # For robustness prefer nltk punkt or syntok if available.

def score_text(text: str):
    sents = split_sentences(text)
    rows = []
    for s in sents:
        wc = len(s.split())
        fk = textstat.flesch_kincaid_grade(s)
        rows.append({
            "sentence": s,
            "words": wc,
            "fk": round(fk, 2),
            "over_ceiling": fk > CEILING,
            "short_flag": wc < MIN_WORDS,   # FK unreliable on very short sentences
        })
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
        "pct_in_band": round(pct_in_band, 2),
        "whole_passage_fk": round(textstat.flesch_kincaid_grade(text), 2),
        "readability_pass": readability_pass,  # HALF the spec — accuracy is separate
        "sentences": rows,
    }
```

**`readability_pass` is only half the spec.** A text can pass FK and still be wrong or vacuous. Accuracy is scored separately (below), and `overall_pass = readability_pass AND accuracy_pass`.

## Accuracy scoring (the other half)

For each output, score accuracy 0/1/2:
- **2** = correct AND explains the real mechanism/cause
- **1** = correct but only a surface definition, no mechanism
- **0** = contains a factual error or misleading oversimplification

`accuracy_pass = (score == 2)` (the spec demands mechanism, not just correctness).

Judge either **manually** (fastest, most trustworthy for ~48 outputs) or with an **LLM judge that is NOT one of the tested models** (avoid self-enhancement bias per MT-Bench). If automating, feed the judge the concept + output + the 0/1/2 rubric and ask for score + one-line justification.

## Ingesting the pasted (non-API) outputs

Create `litmus/manual_outputs.json` for the browser + Qwen outputs; the harness fills GPT via API automatically. Schema:

```json
{
  "gemini":        {"Why is the sky blue?": "…output…", "How do magnets work?": "…"},
  "claude_browser":{"Why is the sky blue?": "…"},
  "qwen_1.7b":     {"Why is the sky blue?": "…"}
}
```

Then score every (model, concept) with `score_text` + the accuracy judgment.

## Output: results table + aggregate

Per (model, concept): `max_fk`, `n_over_ceiling`, `pct_in_band`, `readability_pass`, `accuracy_score`, `overall_pass`, and a one-line **failure-mode note** (e.g. "drifts to grade 6 by sentence 3", "grade 3 but skipped the mechanism").

Then per model, the headline number:
> **Prompted {model} passed the full spec on N/12 concepts. Dominant failure mode: {…}.**

Save results to `litmus/results.json` and print a markdown table. This is the artifact that becomes DOK 1 evidence.

## Optional: consistency probe (the "every time" criterion)

For 2–3 concepts, run each model 3× and report FK variance. Prompting failing *on average* is good evidence; prompting being *inconsistent run-to-run* is even stronger, since the spec demands reliability every time.

## Limitations to record (for honest error analysis)

- Browser frontier temps aren't controllable → cross-model comparison isn't perfectly matched.
- FK is a surface metric (length only); short-sentence scores are noisy — hence `short_flag` and the paired accuracy judge.
- Single run per concept (unless the consistency probe is run) measures breadth, not run-to-run stability.
