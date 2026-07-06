# SmallLearningModel

Fine-tuning a small open base model into a reliable behavioral specialist via QLoRA — see `Train Your Own Small Learning Model.md` for the full assignment brief.

**Status:** environment scaffold only. Behavior direction, Behavior Spec, and eval harness are not yet chosen/built — do not start training until they exist.

## Structure

- `data/` — dataset generation scripts + versioned dataset outputs (the dataset is the real deliverable)
- `eval/` — eval harness + results, built *before* training
- `train/` — QLoRA training notebook (Colab GPU)
- `scripts/smoke_test.py` — local CPU inference check that the environment works
- `requirements.txt` — pinned local dev/inference dependencies

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
2. Behavior Spec + eval harness + data-gen pipeline + smoke test (50 junk examples)
3. v1 dataset + first real training run + first base-vs-tuned eval
4. v2 dataset iteration on a diagnosed failure mode
5. Final eval, error analysis, demo, ship

**Rule:** no training before the eval harness exists.
