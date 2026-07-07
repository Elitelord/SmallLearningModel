# litmus/ — Litmus test + reusable FK-scoring eval harness

Answers the assignment's litmus question — **"can a well-prompted model already
do this reliably?"** — for the Behavior Spec, across four models, and leaves
behind the FK-scoring harness we reuse all week (same ruler as the Day-3 eval).

## Behavior Spec under test

> Given any elementary science concept, produce an explanation where **no
> sentence exceeds Flesch-Kincaid grade 3.0**, **≥70% of sentences fall in FK
> 2.0–3.0**, and it **stays factually correct and conveys the core mechanism**
> (not just a definition).

`overall_pass = readability_pass AND accuracy_pass`, where
`accuracy_pass = (accuracy_score == 2)`.

## Files

| File | Role |
|---|---|
| `concepts.py` | The 12 frozen concepts + the verbatim eval prompt (reuse all week) |
| `fk_score.py` | Per-sentence Flesch-Kincaid scorer — the reusable ruler |
| `accuracy.py` | 0/1/2 accuracy rubric: manual loader + optional (biased) LLM judge |
| `run_api.py` | Generate GPT outputs via the OpenAI API |
| `run_qwen.py` | Generate Qwen3 outputs locally on CPU (the tune target) |
| `score_all.py` | Combine outputs + FK + accuracy → `results.json` + `results.md` |
| `env.py` | Minimal `.env` loader |
| `outputs/*.json` | Raw generations from the runners (one file per model) |
| `manual_outputs.json` | Paste target for browser models (Gemini, Claude) |
| `accuracy_scores.json` | 0/1/2 accuracy judgments per (model, concept) |
| `results.json` / `results.md` | The scored table — DOK 1 evidence |

## Workflow

All commands run from the repo root with the project venv.

**1. Generate automated outputs**

```
.venv\Scripts\python -m litmus.run_api  --model gpt-4o --temperature 0.7 --key gpt
.venv\Scripts\python -m litmus.run_qwen --model Qwen/Qwen3-0.6B --key qwen_0.6b
```

(`run_qwen` also takes `--model Qwen/Qwen3-1.7B --key qwen_1.7b`.)

**2. Paste the browser outputs** (Gemini + Claude have no API key here — manual)

Paste the eval prompt (`concepts.PROMPT_TEMPLATE` with each concept) into a
fresh Gemini and a fresh Claude browser session, one concept per turn, and drop
each answer into `manual_outputs.json`:

```json
{
  "gemini":         {"Why is the sky blue?": "…output…", "...": "..."},
  "claude_browser": {"Why is the sky blue?": "…output…", "...": "..."}
}
```

Empty strings are skipped, so you can score partway.

**3. Score accuracy** (the half FK can't measure)

Fill `accuracy_scores.json` by hand using the 0/1/2 rubric in `accuracy.py`
(2 = correct + real mechanism, 1 = correct but surface, 0 = wrong/misleading).
Manual is fastest and most trustworthy for ~48 outputs.

> **Judge independence (MT-Bench self-enhancement bias):** the judge must not be
> the model being judged. The only API key here is OpenAI, and GPT is under
> test, so an OpenAI judge would bias the GPT column — don't use it there.
> Likewise, Claude must not judge the `claude_browser` column. `accuracy.py`
> exposes `judge_with_openai()` as a convenience but tags biased cells.

**4. Aggregate**

```
.venv\Scripts\python -m litmus.score_all
```

Writes `results.json` and prints/writes `results.md` with a per-model headline:

> **Prompted {model} passed the full spec on N/12 concepts. Dominant failure
> mode: {…}.**

## Optional: consistency probe (the "every time" criterion)

Run a model 3× on 2–3 concepts and compare FK variance. Prompting failing *on
average* is good evidence; being *inconsistent run-to-run* is stronger, since
the spec demands reliability every time. Re-run `run_qwen`/`run_api` with
different `--key` suffixes (e.g. `qwen_0.6b_run2`) and diff `max_fk`.

## Limitations (honest error analysis)

- Browser frontier temperatures aren't controllable → cross-model comparison
  isn't perfectly matched (recorded as `temperature: uncontrolled`).
- FK is a surface metric (sentence length + syllables only); scores on very
  short sentences are noisy — hence `short_flag` / `n_short_flag` and the paired
  accuracy judge.
- Single run per concept (unless the consistency probe is run) measures breadth,
  not run-to-run stability.
- Sentence splitting defaults to a punctuation regex (reproducible, no corpora);
  nltk punkt is used only if already installed locally.
