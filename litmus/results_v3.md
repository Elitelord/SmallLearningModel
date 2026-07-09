# Litmus baseline under v3 gate + audience-calibrated judge

Same 12 elementary-science concepts, same eval prompt. Readability scored under the **v3 gate** (whole-passage FK 1.5-3.0, per-sentence FK std-dev <=1.3, no >=10-word sentence >4.0); accuracy 0/1/2 via gpt-4o (audience-calibrated). Overall pass = readability_v3 AND accuracy==2.

| Model | readability (v3) | accuracy=2 | overall pass |
|---|---|---|---|
| gpt-4o | 1/12 | 12/12 | 1/12 |
| gemini | 0/12 | 12/12 | 0/12 |
| claude_browser | 1/12 | 12/12 | 1/12 |
| Qwen/Qwen3-0.6B | 0/12 | 5/12 | 0/12 |
| Qwen/Qwen3-4B | 5/12 | 12/12 | 5/12 |

## Capacity gain: 0.6B -> 4B

- **Accuracy floor CLEARED.** Qwen3-4B scores accuracy=2 on **12/12** (base, pre-tuning), up from 0.6B's **5/12** (+7) — now matching every frontier model. The 0.6B errors (like-poles attracting, moon phases from rotation, warm water holding more oxygen) are gone.
- **Readability is still the binding gap.** 4B passes the v3 readability gate on only 5/12 (vs frontier 0-1/12) — better than frontier because non-thinking Qwen writes tersely, but 5/12 is not reliable. One failure ('why is the sky blue?') was too SIMPLE (wpFK 0.56, below the 1.5 floor), the rest too hard.
- **So 4B base = 5/12 overall.** That is the bar the QLoRA fine-tune must beat: push readability/overall from 5/12 toward reliable (~12/12) while holding accuracy at 12/12. The thesis is unchanged — no prompted model hits the spec *reliably*; the tune must.

*Method note: Qwen runs use the existing harness (temperature 0.7, thinking disabled, max_new_tokens 320) on CPU. Frontier + 0.6B rows are the saved litmus outputs re-scored under v3 (not re-run).*