# Final Submission Checklist

## Selected Result

Submit **v4r8**, not v4r9.

| Run | Readability | Accuracy-v2 | Overall-v2 | Decision |
|---|---:|---:|---:|---|
| v4r8 | 9/12 | 9/12 | **8/12** | Final model |
| v4r9 | 4/12 | 8/12 | **4/12** | Rejected 1.5-epoch ablation |

The well-prompted Qwen3-4B baseline is 2/12 overall, so v4r8 demonstrates a
six-pass improvement using only the bare `Explain:` prompt.

## Required Deliverables

- [x] Dataset finalized: `data/v4/gold_v4_r7.jsonl` (485 records)
- [x] Dataset card finalized: `data/v4/DATASET_CARD.md`
- [x] Final adapter selected: v4r8
- [x] Eval harness and raw results committed in `litmus/` and `eval/`
- [x] Base-vs-tuned comparison: `eval/model_comparison.md`
- [x] BrainLift evidence and error analysis completed
- [x] Model card prepared: `submission/model/README.md`
- [x] Gradio demo prepared: `demo/`
- [ ] Restore `MyDrive/SmallLearningModel/v4r8/adapter` locally or in Colab
- [x] Publish model and dataset to Hugging Face
- [ ] Launch the Gradio demo in Colab and confirm one response
- [ ] Record and upload the 3-5 minute demo video
- [ ] Insert the final URLs below

## Publish

The local default adapter path is `train/adapters/v4r8`. If publishing from Colab,
pass `/content/drive/MyDrive/SmallLearningModel/v4r8/adapter` instead.

```powershell
$env:HF_TOKEN = "hf_..."
.venv\Scripts\python.exe -m scripts.publish_submission `
  --adapter train/adapters/v4r8 `
  --model-repo YOUR_USERNAME/qwen3-4b-grade3-science-v4r8 `
  --dataset-repo YOUR_USERNAME/grade3-science-explanations-v4r7 `
  --space-repo YOUR_USERNAME/grade3-science-explainer
```

Run first with `--dry-run` to verify that every required file is present.

## Submission URLs

- GitHub: https://github.com/Elitelord/SmallLearningModel
- Dataset: https://huggingface.co/datasets/SAgarwal34/grade3-science-explanations-v4r7
- Model: https://huggingface.co/SAgarwal34/qwen3-4b-grade3-science-v4r8
- Demo: local/Colab Gradio app in `demo/` (public CPU Space requires HF Pro)
- Video: `TODO`
- BrainLift: `Grade-Level Science Explainer — BrainLift.md`

## Launch the Demo in Colab

Hugging Face returned HTTP 402 because hosted Gradio CPU Spaces now require Pro.
The same app can run on the existing Colab T4 and issue a temporary public
`gradio.live` URL:

```python
!git clone https://github.com/Elitelord/SmallLearningModel.git
%cd SmallLearningModel
!pip install -q -r demo/requirements.txt bitsandbytes
%env MODEL_ID=SAgarwal34/qwen3-4b-grade3-science-v4r8
%env GRADIO_SHARE=1
!python demo/app.py
```

Keep the cell running while recording the demo or while the submission is reviewed.

## Demo Video Outline

**0:00-0:35 — Behavior and litmus.** State the falsifiable spec and show that the
well-prompted Qwen3-4B base passes only 2/12 overall.

Show `README.md` → **Behavior Spec**, then `eval/model_comparison.md` →
**Accuracy-v2 Multi-Judge Results / Headline**.

**0:35-1:20 — Dataset.** Show the 485-record clean union, its three replay components,
strict readability target, cross-family accuracy gate, deduplication, and reserved
prompt exclusions.

Show `data/v4/DATASET_CARD.md` → **Composition**, **Quality Gates**, and
**Evaluation Separation**. Briefly open `data/v4/gold_v4_r7.stats.json` to show the
485-record count and source fingerprints.

**1:20-2:05 — Training.** Show assistant-only masking and the final v4r8 recipe:
Qwen3-4B, QLoRA r32/a64, two epochs, `2e-4`.

Show `train/qlora_train.py` → `build_label_masked_examples`, then `README.md` →
**Final Result**. The notebook's **Controlled v4r8 two-epoch midpoint** section is
optional visual evidence of the completed T4 run.

**2:05-3:20 — Live demo.** Ask “Why is the sky blue?”, “How do magnets work?”, and
“What makes a rainbow?” Show that the app receives only `Explain: ...` and displays
the deterministic readability metrics.

Show the live `gradio.live` page. Keep the Colab app-launch cell running throughout
this section. Use `data/sft_format.py` only if you want to show the literal
`MINIMAL_PROMPT = "Explain: {concept}"` source.

**3:20-4:15 — Evidence.** Show 2/12 base versus 8/12 tuned overall. Explain that v4r7
reached 7/12, v4r8 reached 8/12, and the controlled v4r9 reduction regressed to 4/12.

Show `eval/model_comparison.md` → **Headline**, **Tuned Iteration Comparison**, and
**Final Selection: v4r8**. The final blind evidence is recorded in
`eval/FINAL_BLIND_RUN.md` after the v4r8 blind output is judged.

**4:15-4:45 — Honest limitations.** Mention the remaining seasons, plants, lungs, and
moon failure modes and that formula readability is not direct child-comprehension
testing.

Show `Grade-Level Science Explainer — BrainLift.md` →
**Did data → behavior hold? (Evidence)**, or the **Limitations** section on the
published Hugging Face model card.

## Recording Source Map

| What to show | File or URL |
|---|---|
| Behavior spec | `README.md` |
| Base/frontier/tuned table | `eval/model_comparison.md` |
| Dataset composition and gates | `data/v4/DATASET_CARD.md` |
| Frozen dataset metadata | `data/v4/gold_v4_r7.stats.json` |
| Assistant-only masking | `train/qlora_train.py` |
| Minimal inference prompt | `data/sft_format.py` |
| BrainLift conclusion | `Grade-Level Science Explainer — BrainLift.md` |
| Final blind protocol/results | `eval/FINAL_BLIND_RUN.md` |
| Published model | https://huggingface.co/SAgarwal34/qwen3-4b-grade3-science-v4r8 |
| Published dataset | https://huggingface.co/datasets/SAgarwal34/grade3-science-explanations-v4r7 |
