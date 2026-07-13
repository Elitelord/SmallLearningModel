# Grade-Level Science Explainer

A QLoRA adapter that teaches Qwen3-4B to explain elementary science at a
third-grade reading level from the deliberately minimal prompt `Explain: {concept}`.
The project tests whether a narrow behavior can be placed in model weights through
carefully gated data rather than recovered through a long prompt.

## Behavior Spec

Given an elementary physical- or life-science concept, produce a correct explanation
whose whole-passage Flesch-Kincaid grade is 3.0-6.0, ARI is 3.0-7.0, sentence-level
FK standard deviation is at most 1.7, and no sentence of at least ten words exceeds
FK 8.0. Accuracy-v2 requires factuality at least 2/3 and a complete mechanism score
of 2/2.

## Final Result

v4r8 is the selected final adapter. It uses Qwen3-4B, the 485-record v4r7 dataset,
QLoRA r32/alpha64, two epochs, learning rate `2e-4`, batch size 8, and gradient
accumulation 2. Decode temperature 0 was selected on a separate 24-prompt calibration
set before the fixed 12-prompt development litmus was run.

| Model | Prompt | Readability | Accuracy-v2 | Overall-v2 |
|---|---|---:|---:|---:|
| Qwen3-4B base | full grade-3 prompt | 2/12 | 9/12 | **2/12** |
| Best frontier-v2 baseline | full grade-3 prompt | 4/12 | 12/12 | **4/12** |
| **Qwen3-4B + v4r8** | bare `Explain:` | **9/12** | 9/12 | **8/12** |

The adapter gains six joint passes over the well-prompted base while using a weaker
prompt. The final 1.5-epoch v4r9 ablation regressed to 4/12 overall, so v4r8 remains
the evidence-backed selection. Full results and error analysis are in
`eval/model_comparison.md`.

## Final Artifacts

- Published model: https://huggingface.co/SAgarwal34/qwen3-4b-grade3-science-v4r8
- Published dataset: https://huggingface.co/datasets/SAgarwal34/grade3-science-explanations-v4r7
- Training dataset: `data/v4/gold_v4_r7.jsonl`
- Dataset metadata: `data/v4/gold_v4_r7.stats.json`
- Training notebook: `train/train_qlora.ipynb`
- Final judged outputs: `eval/v4r8_decode_litmus_accuracy_v2.json`
- Eval report: `eval/model_comparison.md`
- BrainLift: `Grade-Level Science Explainer — BrainLift.md`
- Hugging Face publisher: `scripts/publish_submission.py`
- Gradio demo: `demo/app.py`
- Submission checklist and video outline: `SUBMISSION.md`

The adapter weights are intentionally excluded from Git. Restore v4r8 from
`MyDrive/SmallLearningModel/v4r8/adapter` or download the published adapter from the
Hugging Face model repository.

## Reproduce the Evaluation

```powershell
.venv\Scripts\python.exe -m litmus.judge_accuracy_v2 `
  --input eval/v4r8_decode_litmus.json `
  --out eval/v4r8_decode_litmus_accuracy_v2.json `
  --resume

.venv\Scripts\python.exe -m eval.summarize_sweep_v2 `
  eval/v4r8_decode_litmus_accuracy_v2.json
```

Accuracy-v2 uses GPT-5.4 and Claude Opus 4.7 as primary judges and Gemini 3.1 Pro
only when their factuality or mechanism scores differ. Consensus is the per-axis
median. Readability is deterministic.

## Run the Demo

Set `MODEL_ID` to the published v4r8 adapter repository and run:

```powershell
uv pip install --python .venv -r demo/requirements.txt
$env:MODEL_ID = "YOUR_USERNAME/qwen3-4b-grade3-science-v4r8"
.venv\Scripts\python.exe demo/app.py
```

The app uses the same bare prompt and deterministic decoding as the final evaluation.

## Repository Map

- `data/` - generation, replay, audit, and the final 485-record dataset
- `train/` - assistant-only label masking, QLoRA trainer, and Colab notebook
- `litmus/` - fixed concepts, readability gate, and multi-judge accuracy-v2
- `eval/` - decode calibration, saved model outputs, summaries, and comparisons
- `demo/` - Hugging Face Spaces-compatible Gradio app
- `submission/` - model card uploaded with the adapter
- `tests/` - masking, data-policy, eval-policy, and judge-consensus tests

## Local Setup

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GPU training uses Unsloth in `train/train_qlora.ipynb`. The public inference prompt
remains byte-for-byte aligned with training through `data/sft_format.py`.
