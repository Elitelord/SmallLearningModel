# **Grade-Level Science Explainer — BrainLift**

*Train Your Own Small Learning Model — one-week build. DOK tiers and the "did data→behavior hold?" evidence section fill in as the week progresses. Tune target: **Qwen3-4B Instruct** (upgraded Day 3 from Qwen3-0.6B — the 0.6B cleared readability but floored on accuracy at 5/12; 4B hits 12/12 accuracy, isolating readability as the behavior to train).*

## **Owners**

- Sameer

## **Purpose**

### **Core Goal**

Instill **one** reliable behavior into a small open base model (tune target: **Qwen3-4B Instruct**) via QLoRA supervised fine-tuning: given any elementary physical- or life-science concept, explain it so that the whole explanation **reads at a third-grade level** while **remaining factually correct and conveying the core scientific mechanism**. The goal is not to make the model smarter than a frontier model, but to be better at maintaining readability in answers than the well-prompted base model and frontier models.  

### **Target Behavior (Behavior Spec — v4 gate)**

**Given any elementary physical- or life-science concept, the model produces an explanation that reads at a third-grade level and remains factually correct while conveying the core scientific mechanism (not merely a surface definition).** Readability is gated on the **whole passage**, not per sentence (per-sentence FK amplifies syllable-count noise on short sentences):

- Whole-passage Flesch-Kincaid grade within **3.0–6.0** — the band where *genuine* grade-3 reading material sits (see below). Floor 3.0 rejects sub-grade-3 baby-talk; ceiling 6.0 keeps it grade-3, not middle-school.  
- Whole-passage **ARI (Automated Readability Index) within 3.0–7.0** as a second gate. On independent grade labels, ARI tied FK as the best separator; being character-based, it fails *differently* from syllable-based FK and catches passages one of them misjudges.  
- Dispersion: standard deviation of per-sentence FK **≤ 1.7** — enforces even grade-3 texture, catching passages that average fine but lurch between fragments and dense clauses.  
- Per-sentence backstop: any sentence of **≥10 words** must be FK **≤ 8.0** (short sentences exempt as noisy; catches a single genuinely-hard long sentence blowing past the band).  
- Accuracy: correct and conveys the mechanism.

**Forbidden failure:** whole-passage FK outside 3.0–6.0; whole-passage ARI outside 3.0–7.0; per-sentence FK dispersion over the cap; any ≥10-word sentence over FK 8.0; or any factual error / oversimplification that misrepresents the mechanism.

### **In Scope**

- Elementary physical-science and life-science concepts for third graders(the narrow target domain).  
- Whole-passage readability control (v4 gate: whole-passage FK 3.0–6.0 **and** ARI 3.0–7.0  per-sentence FK dispersion cap ≤1.7  length-filtered per-sentence backstop). See Behavior Spec and Categories 1.3–1.4.  
- Preserving factual accuracy and the core mechanism under the readability constraint (the competing-constraint tension).  
- Distillation of training data from a teacher model with a readability-forcing rewrite loop  strict accuracy gate. (Caveat: the teacher was switched from the API model to the in-session model (Opus) mid-pipeline; the setup was not held constant — see Category 1.3 method caveat.)  
- An eval harness built *before* training: FK gate  LLM-as-judge for accuracy  base-vs-tuned comparison (built for the litmus test; reused all week).



### **Out of Scope**

- Domains beyond elementary physical/life science (no "any topic" generality — avoids the mushy-model trap).  
- Reading levels other than the third-grade target band.  
- Pretraining, RLHF from scratch, or architecture changes (SFT/QLoRA only; DPO is a stretch goal).  
- Decode-time control of readability (this project is behavior-from-data; decoding is a separate lens).  
- Beating frontier models on raw capability or trivia benchmarks.  
- Designing and validating a novel readability metric from scratch (out of scope for one week; "FK's variables are insufficient" is argued as a POV, not built).



## **DOK 4 — SPOVs (Spiky Points of View)**

*Strong, defensible opinions a peer could disagree with; each anchored to the target behavior and the project's evidence.*

### **SPOV 1**

Any Deterministic Readability Test like FK and ARI  needs to be combined with an LLM-as-judge accuracy test to evaluate true educational value.  
The deterministic metrics tend to miss cases where a sentence is overly simplistic but fails to actually accomplish it's goal. These don't measure accuracy or the actual concept explanation effectiveness.  That's why LLM as a judge methodology is needed as well, not to prove readability, which it evidently can't, but to prove accuracy, which it does. 

### **SPOV 2**

A teacher model doesn't necessarily have to be initially good at the task it's teaching.  
As shown in evidence, frontier models aren't good at readability, but with a deteministic metric filter + rewrite loop, one of those models can still generate a viable dataset, especially when combined with authentic human generated data for reference. 

### **SPOV 3**

You cannot validate a deterministic metric on model-generated data tuned to that metric.  
You need some level of human generated input to prevent circular training against a gate. In this project, the FK band looked correct because it was being tested against data tuned to it. That leads to overfitting, a common issue in many pipelines. A gate is only trustworthy once it is checked against an independent truth. In this case that was the CommonLit CLEAR corpus, graded by human teacher judgment rather than any formula. 

### **SPOV 4**

For a single behavior, base-model capacity is as important as the fine-tuning.  
No amount of QLoRA tuning on Qwen3-0.6B would have fixed its accuracy issues, only getting 5/12. Swapping to Qwen3-4B was needed to get it to 12/12. The initial model needs some base capability and can then be fine-tuned off of. The dataset can then instill specific behaviors that are desired by the SLM. 

## **Experts**



### **Tim Dettmers**

- **Who:** Researcher; lead author of QLoRA.  
- **Focus:** Efficient fine-tuning and quantization — 4-bit QLoRA that makes single-GPU SFT of small models practical.  
- **Why Follow:** The method your entire training step rests on; his framing of "the model is your data made runnable on cheap hardware" is the technical backbone of the build.  
- **Where:** [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)



### **Daniel Han (Unsloth)**

- **Who:** Co-creator of Unsloth, the recommended QLoRA framework for this assignment.  
- **Focus:** Fast, low-VRAM fine-tuning of small open models; clean single-GPU notebooks.  
- **Why Follow:** The practical toolchain for your training runs — 2× faster, 70% less VRAM, which matters for a one-week loop.  
- **Where:** [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)



### **Rudolf Flesch & J. Peter Kincaid**

- **Who:** Originators of the Flesch Reading Ease and Flesch-Kincaid Grade Level formulas.  
- **Focus:** Quantifying readability from sentence length and syllable counts.  
- **Why Follow:** FK grade *is* your measurable constraint and the deterministic half of your eval. Its blind spots on short sentences directly drove the v3 gate redesign.  
- **Where:** [Flesch–Kincaid readability tests](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)



## **DOK 3 — Insights**

*Synthesized, cross-source insights — the "so what" that connects facts into understanding.*

### **From Readability Measurement**

Flesch-Kincaid grade is computed purely from surface features — syllables-per-word and words-per-sentence — so it is blind to word familiarity, real syntactic complexity, and whether the content is even correct. A passage can hit grade 3 with choppy fragments, or with simple-sounding but factually wrong sentences.

### **From QLoRA / SFT Mechanics**

Because QLoRA makes single-GPU fine-tuning cheap and the tooling (Unsloth / TRL / PEFT) is commodity, compute and method are *not* where quality comes from — the dataset is the entire lever. QLoRA's own finding reinforces this: a small, high-quality dataset beat larger prior state-of-the-art.

### **From the Litmus Baseline**

Every capable model tested hits 12/12 on accuracy (GPT-4o, Gemini, Claude, and — after the 0.6B→4B upgrade — Qwen3-4B) yet none reliably meets the readability gate. This holds across every version of the gate: under the original per-sentence spec the best was 1/12; re-scored under the recalibrated v4 grade-3 band (FK 3–6 + ARI + evenness) the best is still only 4/12 (Gemini). This proves they have the knowledge, just not the reliable *ability to hold the readability constraint* — which is exactly the behavior worth training. Notably, the binding failure under v4 is **unevenness** (5–8 of 12 outputs sit in the grade-3 band on average but lurch sentence-to-sentence), and models miss in opposite directions — Qwen3-4B too simple (below the floor), Gemini too hard (over the ceiling).

### **From the Gate-Design Experiment**

A floor/dispersion grid on 37 hand-authored grade-3 explanations revealed that FK rates genuinely clear, correct grade-3 prose at whole-passage 1.3–2.0 — *below* the 2.0–3.0 band the original spec assumed. Forcing the average up to a 2.0 floor pushes the author to over-enrich sentences, which then spike the ceiling — i.e., the floor was *manufacturing* the ceiling violations, not preventing baby-talk. Realization: "readability" is better modeled as central tendency (is it grade-3 on average?) *plus* dispersion (is it evenly grade-3, or lumpy?) than as a single per-sentence pass/fail. This is why the gate uses a whole-passage band  a dispersion cap  a length-filtered backstop, keeping the per-sentence band only as a diagnostic.

### **From the Independent Ground-Truth Recalibration**

The gate-design experiment above was still calibrated on *our own* AI-authored text that had been tuned toward FK — so "is this the right band?" was answered circularly. Testing against an **independent** corpus (CommonLit CLEAR, whose grade labels come from teacher pairwise judgments, not any formula) flipped a key assumption: genuine grade-3 informational prose scores whole-passage FK ≈ 5.5, not ≤ 3.0 — only 5.9% of real grade-3 text passes FK ≤ 3.0. The v3 band (1.5–3.0) had been unknowingly targeting ~grade 1–2. Two further findings fell out: (1) it's a *vocabulary* effect, not sentence length — real grade-3 text and our v3 gold share ~11 words/sentence, so the difference is syllables-per-word; and (2) on those honest labels FK and ARI are the two best difficulty separators (AUC ≈ 0.99, tied), while Dale-Chall — the metric hypothesized to catch FK's blind spot — is the *worst* (AUC 0.66) and an LLM-as-judge for *readability* was badly miscalibrated (rated 23/24 adult passages "easy for an 8-year-old"). Net: keep FK (add ARI), move the band up to real grade 3 (v4: FK 3–6), and don't trust a metric's *absolute* grade number — the formulas rank difficulty well but read plain prose ~2–3 grades high.

## **DOK 2 — Knowledge Tree**

*Categories → Subcategories → Sources. Each source carries its own DOK 1 Facts (atomic, verifiable) and a DOK 2 Summary (one-line synthesis).*

### **Category 1 — Readability Measurement (FK and its limits)**



#### **Subcategory 1.1 — The Flesch-Kincaid Grade Formula**

- [Flesch–Kincaid readability tests (overview  formula)](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)  
  - DOK 1 Facts:  
    - Two related tests exist: the Flesch Reading-Ease and the Flesch–Kincaid Grade Level.  
    - Both use the same core inputs — word length and sentence length — but with different weighting factors.  
    - The two scores correlate approximately inversely: high Reading-Ease ≈ low Grade-Level.  
    - Flesch devised Reading-Ease; he and J. Peter Kincaid later developed the Grade Level formula for the U.S. Navy.  
    - In Reading-Ease, higher scores  easier to read; lower  harder.  
    - Reading-Ease formula: 206.835 − 1.015 × (total words / total sentences) − 84.6 × (total syllables / total words).
  - Summary: Two tests to determine difficulty of passage, different weighted factors with inverse correlation.



#### **Subcategory 1.2 — Computing FK in Code & Formula Blind Spots**

- [textstat (Python readability library)](https://github.com/textstat/textstat)  
  - DOK 1 Facts:  
    - textstat.fleschreadingease(text) returns the Flesch Reading-Ease score.  
    - Reading-Ease maxes around 121.22 and has no lower bound — negative scores are valid.  
    - Reading-Ease bands: 90–100 very easy; 60–69 standard; 0–29 very confusing.  
    - textstat.fleschkincaidgrade(text) returns the FK grade; e.g. 9.3 means a 9th-grader could read it.
  - Summary: There are codebases with functions to get scores and grade levels given a piece of text, acting as a deterministic measurement to grade these texts.



#### **Subcategory 1.3 — Gate-Design Experiment (this project — primary research)**

- Floor / dispersion calibration on 37 hand-authored grade-3 explanations gate design, this project  
  - DOK 1 Facts:  
    - On 37 hand-authored explanations that read at grade level and were judged correct: 2/37 passed the original per-sentence spec (no sentence 3.0  ≥70% in 2.0–3.0); 9/37 passed a ceiling-only rule (no sentence 3.0); 32/37 passed whole-passage FK ≤ 3.0.  
    - textstat rates genuinely clear grade-3 prose at whole-passage FK ≈ 1.3–2.0 — below the 2.0–3.0 band the original spec assumed.  
    - Floor/dispersion yield grid: floor 2.0  dispersion ≤1.0 → 17%; floor 1.8  ≤1.3 → 30%; floor 1.5  ≤1.3 → 38%; floor 1.5  ≤1.5 → 40%.  
    - Adopted v3 working values: whole-passage FK 1.5–3.0; per-sentence FK dispersion ≤ 1.3; ≥10-word sentences ≤ 4.0; band 2.0–3.0 as diagnostic. **(SUPERSEDED — the band was recalibrated to the v4 gate in 1.4: FK 3.0–6.0 AND ARI 3.0–7.0, dispersion ≤ 1.7, ≥10-word backstop ≤ 8.0. The *method* below still stands; only the numbers moved.)**  
    - Method caveat: these explanations were authored by the teacher model running inside the Claude Code session (Opus) after a mid-pipeline switch from the API teacher; the teacher setup was not held constant, so any "even a strong model struggles" reading is confounded and should be re-run under a fixed teacher before being leaned on.
  - Summary: The original per-sentence rule was gating on FK's short-sentence noise rather than real difficulty, so readability was remodeled as whole-passage level *plus* dispersion; the specific band this produced was later recalibrated upward against independent data (see 1.4).



#### **Subcategory 1.4 — Independent Ground-Truth Recalibration (CommonLit CLEAR — primary research)**

- Metric re-test against a formula-independent grade-labeled corpus this project; `eval/build_grade3_real.py`, `eval/run_grade3_real.py`, `eval/metric_comparison_real.md`  
  - DOK 1 Facts:  
    - **CommonLit CLEAR corpus:** 4,724 reading excerpts, each carrying a `Lexile Band` (grade designation) and `BT_easiness` — a Bradley-Terry ease score derived from *teacher pairwise judgments*, not any readability formula. This makes it an independent ground truth; the project's earlier eval set (`fk_eval_drafts_37.jsonl`) had AI-authored positives iterated against FK, so it was circular.  
    - Test set built: 152 real grade-3 positives (Info category, Lexile band 500/700 ≈ grade 2–4) vs 152 adult negatives (band 1300/1500 ≈ grade 9–12), balanced; all metrics recomputed with the repo's textstat 0.7.13.  
    - **Genuine grade-3 informational prose averages whole-passage FK ≈ 5.50 (median 5.16, IQR ~3.5–8); ARI ≈ 5.86; only 5.9% passes FK ≤ 3.0.** Per-Lexile-band gradient is monotonic (band 500 → FK 4.38, band 700 → FK 6.38, band 900 → 8.19 …), and the FK=3 line falls around Lexile band 300 (~grade 1–2).  
    - Confound ruled out: real grade-3 text (11.8 words/sentence) and the project's v3 gold (11.0 w/s) have near-identical sentence length, so the FK gap (5.50 vs 2.35) is driven by syllables-per-word (vocabulary), not choppiness.  
    - On the independent labels, single-metric separation (grade-3 vs adult): **FK AUC 0.990, ARI 0.988** (tied best), SMOG 0.975, Coleman-Liau 0.923, **Dale-Chall 0.876 (worst)**. No AND-combination or leave-one-out-CV logistic beat FK-alone by a meaningful margin (best was FK+ARI, one error fewer); more metrics overfit.  
    - No formula scores grade-3 text *as* grade 3 in absolute terms — with a "grade output in [2,4]" window, FK captured only 21% of real grade-3 passages, ARI 17%, others near 0; they read plain prose ~2–3 grades high.  
    - The gpt-4o LLM-as-judge, asked to rate *readability* on this set, was badly miscalibrated: it rated 23 of 24 adult FK-10+ passages as "easy for an 8-year-old."
  - Summary: Independent labels showed that FK and ARI were best but FK worked better when shifted.



### **Category 2 — QLoRA / SFT Method & Tooling**



#### **Subcategory 2.1 — The Method (LoRA → QLoRA)**

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)  
  - DOK 1 Facts:  
    - LoRA freezes the pretrained weights and injects trainable rank-decomposition matrices into each Transformer layer.  
    - On GPT-3 175B it cuts trainable parameters by up to 10,000× and GPU memory need by 3× vs full fine-tuning.  
    - Matches or beats full fine-tuning quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, with higher training throughput.  
    - Adds no additional inference latency (unlike adapters).
  - Summary: Rather than fine-tuning larger models, freeze weights and inject matrices into each layer, reducing trainable parameters, gpu requirement, and increasing training throughput.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)  
  - DOK 1 Facts:  
    - QLoRA finetunes a 65B model on a single 48GB GPU while preserving 16-bit fine-tuning performance.  
    - It backpropagates gradients through a frozen, 4-bit quantized model into LoRA adapters.  
    - Introduces 4-bit NormalFloat (NF4), a data type information-theoretically optimal for normally-distributed weights.  
    - Adds double quantization (quantizing the quantization constants) to cut memory further.  
    - Adds paged optimizers to manage memory spikes.  
    - Its Guanaco family reached 99.3% of ChatGPT on the Vicuna benchmark with 24h of single-GPU finetuning.  
    - Key finding: QLoRA on a small, high-quality dataset can beat larger prior SoTA — data quality outweighs size.
  - Summary: QLora is a low-memory fine tuning approach that reaches almost same performance with quicker time and lower compute cost. Uses new data type and other methods like quantization and paged optimizers to achieve this.



#### **Subcategory 2.2 — Training Tooling**

- [Unsloth (fast, low-VRAM QLoRA)](https://github.com/unslothai/unsloth)  
  - DOK 1 Facts:  
    - Trains and RLs 500+ models up to 2× faster with up to 70% less VRAM, with no accuracy loss.  
    - Uses custom Triton and mathematical kernels.  
    - Supports full fine-tuning, RL, pretraining, and 4-bit / 16-bit / FP8 training.  
    - Includes data recipes (auto-create datasets from PDF/CSV/DOCX) and live training observability.  
    - Multi-GPU training is supported.
  - Summary: repo that helps users train models quickly on less compute with many features included.
- [TRL (HF supervised fine-tuning / SFTTrainer)](https://github.com/huggingface/trl)  
  - DOK 1 Facts:  
    - Post-training library for SFT, GRPO, and DPO, built on the HF Transformers ecosystem.  
    - Provides ready trainers: SFTTrainer, GRPOTrainer, DPOTrainer, RewardTrainer.  
    - Scales single-GPU → multi-node via Accelerate (DDP, DeepSpeed).  
    - Full PEFT integration for quantization  LoRA/QLoRA; integrates Unsloth kernels.  
    - Has a CLI for fine-tuning without writing code.
  - Summary: Similar to Unsloth, library for model training that incorporates LoRA/QLoRA for efficiency.
- [PEFT (HF parameter-efficient fine-tuning)](https://github.com/huggingface/peft)  
  - DOK 1 Facts:  
    - PEFT fine-tunes only a small number of (extra) parameters instead of all model weights.  
    - This sharply cuts compute and storage cost; recent PEFT methods rival fully fine-tuned performance.  
    - Integrated with Transformers, Diffusers, and Accelerate.
  - Summary: Another fine-tuning strategy that trains only some parameters for cost reduction.



### **Category 3 — Base Models (small, open, Instruct)**



#### **Subcategory 3.1 — Primary Candidate (Qwen3 small; tune target Qwen3-4B Instruct)**

- [Qwen models on Hugging Face (tune target: Qwen3-4B Instruct)](https://huggingface.co/Qwen)  
  - DOK 1 Facts:  
    - Supports switching between a "thinking" mode (reasoning, math, code) and a "non-thinking" mode (efficient dialogue) within one model.  
    - Improved reasoning over QwQ (thinking) and Qwen2.5-Instruct (non-thinking) on math, code, and commonsense logic.  
    - Strong human-preference alignment: creative writing, role-play, multi-turn dialogue, instruction following.  
    - Strong agent/tool-use capability across both modes.  
    - Supports 100+ languages and dialects.  
    - **Tune target upgraded Qwen3-0.6B → Qwen3-4B (Day 3):** the 0.6B base scored accuracy=2 on only 5/12 litmus concepts (real science errors — like-poles attracting, moon phases from rotation), a capacity floor no amount of readability data would fix; Qwen3-4B base hits accuracy=2 on 12/12, matching the frontier models and isolating readability as the sole behavior the fine-tune must instill. Litmus Qwen runs use non-thinking (instruct-style) mode on CPU, temperature 0.7.
  - Summary: Newer Qwen model with stronger reasoning and dialogue capabilities; 4B chosen over 0.6B to clear the accuracy floor.



#### **Subcategory 3.2 — Alternates (Llama 3.2, Gemma 3, SmolLM3)**

- Gemma 3 (Google) — verify exact HF model-card URL, e.g. huggingface.co/google/gemma-3-  
  - DOK 1 Facts:  
    - Family of lightweight open models from Google, built from the same research as the Gemini models.  
    - Gemma 3 is multimodal (text  image in, text out), with open weights for pretrained and instruction-tuned variants.  
    - 128K context window; multilingual across 140+ languages; more size options than previous versions.  
    - Small enough to deploy on laptops, desktops, or private cloud.
  - Summary: Competitor model from google with small sizes for lower compute environments while maintaining multi modal capabilities.



### **Category 4 — Data Generation & Distillation**



#### **Subcategory 4.1 — Distilling / Synthesizing from a Teacher Model**

- [Self-Instruct: Aligning LMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)  
  - DOK 1 Facts:  
    - A framework that bootstraps instruction data from a model's own generations.  
    - Pipeline: generate instructions plus input/output samples, then filter invalid or near-duplicate ones before finetuning.  
    - Applied to vanilla GPT-3, it gave a 33% absolute gain on Super-NaturalInstructions, matching InstructGPT-001.  
    - Provides a nearly annotation-free method for aligning models to instructions.
  - Summary: Framework to improve models by using their outputs, filtering them, and feeding them back in.



#### **Subcategory 4.2 — Quality Filtering Against the Spec**

- Pipeline built for this project (readability-forcing rewrite loop  strict accuracy gate) your writeup  
  - DOK 1 Facts: what the pipeline does: generate accurate explanation → FK-score → rewrite over-ceiling sentences → repeat (capped) → accuracy gate keeps only score 2 Finding: authoring/curating to the gate is expensive, which drove the pivot to a 80-example gold core before scaling. Gate values in Category 1.3.  
  - Summary: A generate → FK-score → rewrite-until-in-band → accuracy-gate loop that keeps only mechanism-correct, in-band passages; because curating to the gate is expensive, the build starts from a small gold core instead of mass generation.



#### **Subcategory 4.3 — External Grade-3 Source Survey (this project — primary research)**

- Time-boxed survey of external reading-material sources as seeds for the rewrite loop `data/source_survey.md`  
  - DOK 1 Facts:  
    - **Nothing passes the readability gate as drop-in data** — every samplable source scored 0–5% under the v3 gate; real published material (even "simplified") sits at grade ~5–12. (Consistent with Category 1.4: even genuine grade-3 material is FK ~5.5, above the old band.)  
    - Recommended seeds: **Simple English Wikipedia** (CC BY-SA 4.0, science-rich, grade ~5–8) as a style/topic seed, and **ARC-Easy** (CC BY-SA 4.0, grade-school science question stems) as a prompt/topic seed — neither usable as reference text, only as material for the rewrite loop.  
    - Rejected: CommonLit CLEAR (authentic prose, grade 3–12, high dispersion — but excellent as an *eval* ground truth, see 1.4), ASSET/WikiLarge (wrong domain), ELI5 (adult register), ReadWorks / Newsela (licensing-blocked).
  - Summary: No open corpus is drop-in grade-3 training data — the usable ones (Simple English Wikipedia, ARC-Easy) serve as style/topic seeds for the rewrite loop, while CLEAR is more valuable as independent eval ground truth (1.4) than as training text.



### **Category 5 — Evaluation**



#### **Subcategory 5.1 — LLM-as-Judge Methodology**

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)  
  - DOK 1 Facts:  
    - Proposes using strong LLMs as judges to evaluate models on open-ended questions.  
    - Identifies judge biases: position, verbosity, and self-enhancement bias, plus limited reasoning ability.  
    - Introduces two benchmarks: MT-Bench (multi-turn) and Chatbot Arena (crowdsourced battles).  
    - Strong judges like GPT-4 reach 80% agreement with humans — matching human-to-human agreement.
  - Summary: Explores using LLMs to judge other LLMs, finding that they are capable of reaching 80% agreement, the same as human.



#### **Subcategory 5.2 — Base-vs-Tuned Experimental Design**

- Your rubric design — see the brief's Appendix A: Spec adherence / Robustness / Task quality / Consistency, scored 0/1/2, report delta. Same-ruler requirement now satisfied: the litmus baseline has been re-scored under both the v3 and the operative v4 gate (`litmus/results_v3.md`, `litmus/results_v4.md`), so base-vs-tuned will use one ruler.  
  - DOK 1 Facts:  
  - Summary: Base and tuned models are scored on one ruler (the v4 readability gate + the accuracy judge) with a 0/1/2 rubric across spec-adherence / robustness / task-quality / consistency; the reported result is the delta the fine-tune adds over the prompted baseline.



### **Category 6 — Prompting Baseline (Litmus Test — primary research)**

*Does a well-prompted model already meet the spec? Own experiment, run Day 1 Establishes the baseline the fine-tune must beat.*

#### **Subcategory 6.1 — Litmus Experiment (this project)**

- Litmus test — 4 models × 12 fixed elementary-science concepts, scored on the then-current spec (per-sentence FK ceiling 3.0, band 2.0–3.0 ≥70%, accuracy 0/1/2). litmus/results.md, litmus/accuracyscores.json  
  - DOK 1 Facts:  
    - Full-spec pass (readability AND accuracy  2): GPT-4o 0/12, Gemini 0/12, Claude 1/12, Qwen3-0.6B 0/12 — 1 of 48 outputs total.  
    - Readability alone (per-sentence FK ceiling  band): GPT 0/12, Gemini 0/12, Claude 1/12, Qwen 0/12. The FK-3.0 ceiling is a near-universal failure.  
    - The over-ceiling sentence is frequently sentence 1, so failure is "never starts at grade 3," not late drift.  
    - Accuracy alone: all three frontier models scored 2 on 12/12; Qwen3-0.6B scored 2 on only 5/12.  
    - Qwen's five hard errors included: like-poles described as attracting; moon phases attributed to the moon's own rotation; warm water framed as holding more oxygen; ice melting described as passing through water vapor.  
    - Percent-of-sentences-in-band typically sat at 0–33% across all models — most sentences were above the band, not merely above the ceiling.  
    - The single passing output across all 48 was Claude on "why do we have day and night?" (max FK 2.85, 80% in band, accuracy 2).  
    - Method limits: browser frontier temperatures were uncontrolled; single run per concept (breadth, not run-to-run consistency); accuracy for the Claude column was judged by GPT-4o (a non-Claude judge) to avoid self-enhancement bias.  
    - **Re-scored under v3 gate** (whole-passage FK 1.5–3.0 + dispersion + backstop; accuracy re-judged by gpt-4o audience-calibrated): overall pass GPT 1/12, Gemini 0/12, Claude 1/12, Qwen3-0.6B 0/12, **Qwen3-4B 5/12** (its terseness fit the low band). `litmus/results_v3.md`.  
    - **Re-scored under v4 gate** (recalibrated grade-3 band FK 3–6 **and** ARI 3–7, dispersion ≤1.7): overall pass **GPT 2/12, Claude 1/12, Gemini 4/12, Qwen3-4B 2/12** (Qwen3-4B drops from 5→2 because its terse, sub-grade-3 vocabulary now falls below the 3.0 floor). Accuracy saturated at 12/12 for all capable models; readability is the sole differentiator, and unevenness (dispersion) is the most common binding failure. `litmus/results_v4.md`.
  - Summary: The prompting baseline shows the knowledge is already present (accuracy saturates at 12/12) but no prompted model — frontier or small — reliably holds the readability constraint, so the target behavior has to be trained in, not prompted.



## **Further Sources / Reading**

- Unsloth docs / example notebooks — verify current URL  
- HF Hub dataset-publishing guide — for shipping the dataset deliverable  
- Any additional readability or edu-NLP references you gather



## **Did data → behavior hold? (Evidence)**

**Yes, but not monotonically.** The final v4r8 adapter turns a bare `Explain:` prompt
into the target behavior on **8/12** fixed development-litmus concepts. The
well-prompted Qwen3-4B baseline reaches only **2/12**, so the adapter adds six full
passes even though it receives a much weaker prompt. The best tested frontier prompt
baseline reaches 4/12. This is evidence that the constrained reading behavior lives in
the trained weights rather than in prompt wording.

| Model | Prompt | Readability | Accuracy-v2 | Overall-v2 |
|---|---|---:|---:|---:|
| Qwen3-4B base | full grade-3 prompt | 2/12 | 9/12 | **2/12** |
| Best frontier-v2 (Claude Opus 4.7) | full grade-3 prompt | 4/12 | 12/12 | **4/12** |
| Qwen3-4B + v4r7 | bare `Explain:` | 10/12 | 9/12 | **7/12** |
| **Qwen3-4B + v4r8 (final)** | bare `Explain:` | **9/12** | 9/12 | **8/12** |
| Qwen3-4B + v4r9 | bare `Explain:` | 4/12 | 8/12 | **4/12** |

The decisive data iteration was the 485-record clean union: 98 v4r2 accuracy anchors,
106 v4r4 readability records, and 281 clean v4r5 targets. Every record passes the
tighter training readability band and a two-family clean 3/2 accuracy gate, with all
reserved prompts excluded. Moving from the conservative r16 v4r6 recipe to r32/a64
on this union raised readability from 5/12 to 10/12. Reducing that recipe from three
epochs to two produced v4r8 and improved the joint score from 7/12 to 8/12.

The remaining failure is the **joint** constraint. v4r8 has nine readable outputs,
but seasons is inaccurate; plants is accurate but narrowly exceeds the readability
dispersion cap; lungs and moon phases fail both. The 1.5-epoch v4r9 ablation regressed
to 4/12 overall, showing that less training is not a free accuracy fix. The result is
therefore a strong but imperfect specialist: the data-to-behavior thesis held by a
large measured margin, while mechanism preservation remains the next data-design
problem.
