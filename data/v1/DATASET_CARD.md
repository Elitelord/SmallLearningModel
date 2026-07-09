# Dataset card — v1 gold core (STUB)

> Small SFT dataset teaching a 0.6B model to explain elementary science at a
> 3rd-grade reading level while keeping the real mechanism. Gold core for the
> first real training run; to be expanded after the base-vs-tuned eval.

## What it is
- **Task:** given `Explain: {question}`, produce a grade-3 explanation that stays
  factually correct and conveys the core mechanism (not just a definition).
- **Size:** 90 examples · 30 distinct concepts · 2–3 phrasings each.
- **Domain:** elementary physical + life science (narrow, on purpose — a 0.6B
  learns form faster than facts).

## How it was generated
- **Teacher:** Claude (agent-authored), each explanation iterated against the FK
  harness until it passed the readability gate. (A GPT teacher was tried and
  rejected: best-of-N gpt-4o clears the gate ~1/40 — it writes too richly.)
- **Phrasings:** gpt-4o-mini (input variety only; no readability constraint).
- **Accuracy judge:** gpt-4o with an audience-calibrated 0/1/2 rubric — different
  model family from the teacher (best defense against self-enhancement bias).
  **Only accuracy==2 kept.**

## Filters (every example passes all)
- **Readability gate v3:** whole-passage FK ∈ [1.5, 3.0] · per-sentence FK
  std-dev ≤ 1.3 (over ≥8-word sentences) · no ≥10-word sentence over FK 4.0.
  `pct_in_band` (2.0–3.0) is recorded as a diagnostic only. *(This replaced the
  original ≥70%-in-2.0–3.0 band, which textstat's short-sentence noise made
  near-unsatisfiable — 2/37 vs 32/37.)*
- **Accuracy == 2** (correct AND real child-level mechanism).
- **Format:** 4–8 sentences.
- **Dedup:** near-identical explanations across different concepts removed
  (SequenceMatcher > 0.85).

## Splits / leakage
- **Eval set:** 24 held-out concepts (the 12 litmus concepts + 12 extra), fully
  excluded from train — concepts *and* their phrasings.
- No train concept or phrasing normalizes onto an eval or few-shot-exemplar concept.

## SFT format
- Minimal prompt `Explain: {concept}` (behavior in the weights, not the prompt);
  identical at train and inference. Qwen3 chat template.

## Known limitations (v1)
- **Cadence uniformity:** explanations are almost all 5 sentences of ~11 words;
  this is partly induced by the v3 gate's narrow window. Risk: the student may
  memorize the skeleton. To diversify on expansion.
- Small (gold core). Domain is deliberately narrow.
- FK is a surface readability proxy; the accuracy judge is a single gpt-4o call.

## License
- TODO (intended for HF Hub). Teacher/judge outputs via OpenAI + Anthropic;
  confirm terms before publishing.
