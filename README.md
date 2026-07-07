# SmallLearningModel

Fine-tuning a small open base model into a reliable behavioral specialist via QLoRA — see `Train Your Own Small Learning Model.md` for the full assignment brief.

**Status:** Day 2 done — Behavior Spec set, eval harness reused from `litmus/`, data-gen pipeline built, and the generate → train → eval loop proven end-to-end on 50 junk examples (`scripts/smoke_e2e.py`). Day 3 generates the real v1 dataset and runs the first GPU training.

**Behavior Spec:** given any elementary physical/life-science concept, produce an explanation where no sentence exceeds Flesch-Kincaid grade 3.0, ≥70% of sentences fall in FK 2.0–3.0, and it stays factually correct while conveying the real mechanism. Tune target: Qwen3-0.6B Instruct.

## Structure

- `litmus/` — reusable eval harness (FK scorer + accuracy judge), built Day 1.5; used as BOTH the data-gen readability/accuracy filters AND the eval metric
- `data/` — data-gen pipeline: `gen_concepts.py`, `exemplars.json`, `generate.py` (generate → readability-rewrite → accuracy-gate), `sft_format.py` (the minimal prompt)
- `eval/` — `base_vs_tuned.py`, the base-vs-tuned harness (reuses litmus)
- `train/` — `qlora_train.py` (Unsloth 4-bit on GPU, peft fallback on CPU) + Colab notebook
- `scripts/smoke_e2e.py` — the Part E end-to-end smoke test (generate → train → eval)
- `scripts/smoke_test.py` — Day-1 local CPU inference check
- `requirements.txt` — pinned deps (local CPU + Colab-only notes)

## End-to-end smoke test (Day 2 checkpoint)

```
.venv\Scripts\python -m scripts.smoke_e2e --n 50 --eval-n 5 --max-steps 10
```

Generates 50 junk examples, trains a LoRA adapter (CPU fallback), evals base vs tuned on
held-out concepts, and prints a delta table. Proves the plumbing before spending on the
real dataset.

## Setup

```
uv venv --python 3.11 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python scripts\smoke_test.py
```

The smoke test loads `Qwen/Qwen3-0.6B` and generates a response on CPU — this is the Day 1 checkpoint (base model runs and responds locally).

For actual QLoRA training, open `train/train_qlora.ipynb` in Colab with a GPU runtime (local machine has no CUDA GPU — Unsloth/bitsandbytes require one).

## Plan

Following the one-week arc in the assignment brief:

1. Setup (done) + research behavior + Brainlift
2. Behavior Spec + eval harness + data-gen pipeline + smoke test (50 junk examples) — **done**
3. v1 dataset + first real training run + first base-vs-tuned eval
4. v2 dataset iteration on a diagnosed failure mode
5. Final eval, error analysis, demo, ship

**Rule:** no training before the eval harness exists.
