---
language:
- en
task_categories:
- text-generation
size_categories:
- n<1K
pretty_name: Grade-Level Science Explanations v4r7
---

# Grade-Level Science Explanations v4r7

The final 485-record supervised fine-tuning dataset for the grade-level science
explainer. Each record maps a unique elementary-science question to a concise,
mechanism-complete explanation. Training uses the minimal prompt
`Explain: {phrasing}` so the reading behavior must be learned from examples rather
than supplied through prompt instructions.

## Files

| File | Records | Purpose |
|---|---:|---|
| `gold_v4_r7.jsonl` | **485** | Final training split used by v4r7, v4r8, and v4r9. |
| `gold_v4_r7.stats.json` | - | Hash, composition, gates, and source fingerprints. |

The normalized SHA-256 of the training JSONL is
`254bc91d469557a1171f657cf8e891e0819b467832888262f16fd1d07c45cd74`.

## Schema

- `concept`: canonical topic metadata
- `phrasing`: exact user-side training question
- `explanation`: assistant target
- `accuracy`: raw judgments and accuracy-v2 consensus
- `teacher`, `rewriters`, `judge`: generation provenance
- `mixture_source`: replay component used in the final union

## Composition

| Component | Records | Role |
|---|---:|---|
| v4r2 clean accuracy anchors | 98 | Preserve strong core mechanisms. |
| v4r4 clean readability replay | 106 | Preserve the strongest earlier style examples. |
| v4r5 clean targets | 281 | Broad, benchmark-preserving topic coverage. |

Teachers were Claude Sonnet 4.6 and Gemini 3.1 Pro, with cross-family rewrites.
Every retained record received clean 3/2 consensus from GPT-5.4 and Claude Opus 4.7;
Gemini 3.1 Pro resolved primary disagreements. Historical litmus-targeted records,
all reserved prompts, and duplicate training phrasings are excluded.

## Quality Gates

Every record passes both the tighter training target and the public v4 evaluation
gate.

Training target:

- whole-passage FK 3.3-5.0
- whole-passage ARI 3.8-6.2
- sentence FK standard deviation at most 1.1
- maximum sentence FK at most 7.0
- 4-6 sentences
- accuracy-v2 clean pass: factuality 3/3 and mechanism 2/2

Observed distribution:

| Metric | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| Whole-passage FK | 3.31 | 4.22 | 4.21 | 5.00 |
| Whole-passage ARI | 3.85 | 5.28 | 5.26 | 6.20 |
| Sentence FK standard deviation | 0.06 | 0.80 | 0.77 | 1.10 |
| Maximum sentence FK | 3.84 | 5.21 | 5.34 | 7.00 |

## Evaluation Separation

The final dataset excludes the original 12 development-litmus prompts, the 24-prompt
calibration set, the sealed `blind_v4r5` set, and prohibited near-neighbor targeted
generation. Calibration chooses decoding settings; the development litmus reports
iteration performance. The sealed blind set was not used to select the final run.

## Intended Use and Limitations

This dataset is intended for supervised tuning of small English language models on a
narrow educational explanation style. It is synthetic and should not be treated as a
factual encyclopedia or a substitute for expert-reviewed curriculum. The multi-judge
gate reduces but does not eliminate scientific errors or judge bias.

Generation used commercial model APIs; users publishing derivatives should confirm
that their use complies with the applicable provider and base-model terms.
