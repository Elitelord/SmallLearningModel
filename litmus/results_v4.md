# Litmus baseline under the v4 gate (FK 3.0-6.0 AND ARI 3.0-7.0, dispersion ≤ 1.7)

<!-- accuracy-v2:start -->
## Accuracy-v2 Multi-Judge Results

Rubric `accuracy_v2` uses factuality 0–3 and mechanism 0–2. A clean pass is 3/2; the benchmark accuracy pass allows a minor localized error (factuality ≥2) but still requires mechanism 2. Overall pass also requires the unchanged v4 readability gate.

Primary judges: `openai-group/gpt-5.4` and `claude-group/claude-opus-4-7`. `gemini-group/gemini-3.1-pro` is called only when either primary axis differs; consensus is the per-axis median.

### Frontier-v2 Panel

These rows rerun the original 12 full grade-3 prompts with the highest accessible
frontier models. GPT-5.6 SOL and Gemini 3.1 Pro use temperature 0. Claude Opus 4.8
uses provider-default decoding because its API rejects the deprecated `temperature`
parameter. `claude-group/claude-fable-5` could not be invoked because the gateway's
AWS Bedrock role lacks Marketplace access.

| Model | Readability | Clean 3/2 | Accuracy-v2 | Overall-v2 | Mean F/M | Gemini tie-break |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 SOL | 3/12 | **12/12** | **12/12** | 3/12 | 3.0/2.0 | 0/12 |
| Claude Opus 4.8 | 3/12 | 11/12 | **12/12** | 3/12 | 2.917/2.0 | 2/12 |
| Gemini 3.1 Pro | 1/12 | 11/12 | **12/12** | 1/12 | 2.917/2.0 | 1/12 |

Raw outputs, judgments, and aggregates are in `litmus/frontier_v2_outputs.json`,
`litmus/frontier_v2_accuracy_v2.json`, `litmus/frontier_v2_summary.json`, and the
newer `litmus/frontier_opus48_*` artifacts.

### Headline

| Model | Prompt | Readability | Clean 3/2 | Accuracy-v2 | Overall-v2 | Mean F/M | Gemini |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-4o | full grade-3 prompt | 2/12 | 12/12 | 12/12 | **2/12** | 3.0/2.0 | 4/12 |
| Claude (browser) | full grade-3 prompt | 1/12 | 11/12 | 12/12 | **1/12** | 2.917/2.0 | 2/12 |
| Gemini (browser) | full grade-3 prompt | 4/12 | 8/12 | 12/12 | **4/12** | 2.667/2.0 | 2/12 |
| GPT-5.6 SOL | full grade-3 prompt | 3/12 | 12/12 | 12/12 | **3/12** | 3.0/2.0 | 0/12 |
| Claude Opus 4.8 | full grade-3 prompt | 3/12 | 11/12 | 12/12 | **3/12** | 2.917/2.0 | 2/12 |
| Gemini 3.1 Pro | full grade-3 prompt | 1/12 | 11/12 | 12/12 | **1/12** | 2.917/2.0 | 1/12 |
| Qwen3-4B (base) | full grade-3 prompt | 2/12 | 4/12 | 9/12 | **2/12** | 2.333/1.75 | 9/12 |
| Qwen3-4B + v6 tune (v4r6) | bare Explain: | 5/12 | 4/12 | **10/12** | **5/12** | 2.167/1.833 | 5/12 |
| Qwen3-4B + v7 tune (v4r7) | bare Explain: | **10/12** | 5/12 | 9/12 | **7/12** | 2.167/1.75 | 6/12 |
| Qwen3-4B + v8 tune (v4r8) | bare Explain: | 9/12 | **6/12** | 9/12 | **8/12** | 2.167/1.667 | 5/12 |

### Tuned Iteration Comparison

| Iteration | Readability | Clean 3/2 | Accuracy-v2 | Overall-v2 | Mean F/M |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B + v2 tune (v4r2) | 5/12 | 7/12 | 9/12 | **4/12** | 2.333/1.667 |
| Qwen3-4B + v3 tune (v4r3) | 8/12 | 4/12 | 7/12 | **5/12** | 2.0/1.5 |
| Qwen3-4B + v4 tune (v4r4) | 9/12 | 3/12 | 7/12 | **5/12** | 1.667/1.333 |
| Qwen3-4B + v5 tune (v4r5) | 7/12 | 3/12 | 8/12 | **3/12** | 1.917/1.583 |
| Qwen3-4B + v6 tune (v4r6) | 5/12 | 4/12 | **10/12** | **5/12** | 2.167/1.833 |
| Qwen3-4B + v7 tune (v4r7) | **10/12** | 5/12 | 9/12 | **7/12** | 2.167/1.75 |
| Qwen3-4B + v8 tune (v4r8) | 9/12 | **6/12** | 9/12 | **8/12** | 2.167/1.667 |
| Qwen3-4B + v9 tune (v4r9) | 4/12 | 4/12 | 8/12 | **4/12** | 2.0/1.5 |

### v4r5 Regression

v4r5 improves tolerant accuracy by one pass versus v4r4, but loses two readability
passes and two overall passes. Its three overall passes are day/night, lungs, and fish.
The clean multi-judge data gate improved training-target quality, but the conservative
r16/two-epoch run did not reliably learn even readability or preserve several core
mechanisms. Raw judgments are in `eval/v4r5_decode_litmus_accuracy_v2.json`. The new
`blind_v4r5` holdout remains unrun.

### v4r6 Mixed-Replay Result

v4r6 combines 98 clean tight v4r2 accuracy anchors, 102 clean tight v4r4
readability records, and 200 clean v4r5 targets. It raises tolerant accuracy to
**10/12**, the strongest tuned-model result, with only rainbow and moon phases failing.
Readability falls to **5/12**, however, so overall-v2 reaches only **5/12**. The five
overall passes are sky, ice, gravity, puddles, and lungs. The accuracy-anchor strategy
worked, but the conservative r16/two-epoch recipe did not retain r4's readability.
Calibration readability (12/24 at temperature 0) also overstated development-litmus
readability, so calibration is useful for decoding selection but not a sufficient
progression proxy by itself. Raw judgments are in
`eval/v4r6_decode_litmus_accuracy_v2.json`; `blind_v4r5` remains sealed.

### v4r7 Capacity-Recipe Result

v4r7 transfers v4r4's r32/a64, three-epoch, `2e-4` recipe to the complete
485-record clean union. Readability rises from 5/12 to a new best **10/12**, while
accuracy-v2 remains **9/12** and overall-v2 reaches a new best **7/12**. The seven
overall passes are sky, plants, day/night, ice, puddles, rainbows, and fish. Magnets,
gravity, and seasons fail accuracy despite passing readability; lungs and moon phases
pass accuracy but miss readability. This result validates the higher-capacity recipe,
and motivates a controlled two-epoch test on the same data and configuration to probe
the accuracy/readability balance. Raw judgments are in
`eval/v4r7_decode_litmus_accuracy_v2.json`; the blind holdout remains sealed.

### v4r8 Two-Epoch Ablation

v4r8 keeps v4r7's dataset and r32/a64, `2e-4` recipe but reduces training from
three epochs to two. It trades one readability pass for one additional clean pass and
raises overall-v2 from **7/12** to a new best **8/12**. Seasons is the only accuracy
failure among its nine readable outputs; plants is fully accurate but narrowly misses
the readability dispersion cap. Lungs and moon phases fail both gates. Raw judgments
are in `eval/v4r8_decode_litmus_accuracy_v2.json`; the blind holdout remains sealed.

### Final Selection: v4r8

v4r9 reduced the same r32/a64 run from two epochs to 1.5. Its calibration-selected
temperature `0.7` produced only 4/12 readability, 8/12 accuracy-v2, and 4/12
overall-v2. The lower training dose therefore underfit both the target style and
several mechanisms. v4r8 remains the final model at **8/12 overall-v2**, a six-pass
gain over the well-prompted Qwen3-4B baseline's 2/12 while using only the bare
`Explain:` prompt. Raw v4r9 judgments are in
`eval/v4r9_decode_litmus_accuracy_v2.json`.

### Judge Agreement

The aggregate agreement figures below cover the original 96-output matrix; later
v4r5-v4r9 and Frontier-v2 rows were judged separately with the same rubric and models.

- Exact two-axis agreement: 56/96 (58.3%).
- Accuracy-pass agreement: 76/96 (79.2%).
- Linear weighted kappa: factuality 0.651, mechanism 0.526.
- Gemini tiebreakers: 40/96.
- Judge-family relationships are recorded per output in the raw JSON; cross-family agreement should be preferred when interpreting tested GPT, Claude, or Gemini rows.

### Per-Concept Detail

#### GPT-4o

| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Why is the sky blue? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains Rayleigh scattering in child-friendly terms, including why blue scatters more (shorter waves) and why sunsets appear red. |
| How do plants make their own food? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly identifies chlorophyll, sunlight, air, and water combining to make food, conveying the core photosynthesis mechanism appropriately for a child. |
| Why do we have day and night? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly identifies Earth's rotation as the cause and clearly conveys the cause-and-effect chain of facing toward or away from the sun. |
| What makes ice melt? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains that heat energy makes water molecules move faster and break apart from their frozen structure, delivering the core mechanism at an age-appropriate level. |
| How do magnets work? | 3/2 | 2/1 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation is completely accurate and provides a clear, age-appropriate mechanism for magnetic attraction involving poles, forces, and magnetic domains. |
| Why do things fall to the ground? | 3/2 | 2/1 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation uses highly effective, age-appropriate analogies to accurately describe gravity as a pulling force generated by Earth's large size. |
| Where does a puddle go when it dries up? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains evaporation via sun-heated molecules gaining energy and escaping into the air as gas, with a clear cause-and-effect chain suitable for a child. |
| Why do we have seasons? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly identifies axial tilt causing varying sunlight to different parts of Earth throughout its yearly orbit, conveying the core mechanism at an age-appropriate level. |
| How do our lungs help us breathe? | 3/2 | 3/1 | 3/2 | 3/2 (5/5) | ✅ | ✅ | ✅ | The explanation correctly and clearly describes the mechanism of gas exchange, including inhaling oxygen, transferring it to the blood, and exhaling carbon dioxide, using highly appropriate language for a 7-year-old. |
| What makes a rainbow? | 3/2 | 2/2 | 3/2 | 3/2 (5/5) | ✅ | ✅ | ✅ | The explanation accurately and simply describes how white sunlight bends and splits into different colors as it passes through raindrops, correctly noting the required position of the sun and rain. |
| Why does the moon look like it changes shape? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains that the moon reflects sunlight and we see different portions of its lit side as it orbits Earth, conveying the core mechanism in child-friendly terms. |
| How do fish breathe underwater? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation correctly describes the gill mechanism—water enters the mouth, flows over gills that extract oxygen into the blood—at an age-appropriate level. |

#### Claude (browser)

| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Why is the sky blue? | 3/2 | 3/2 | — | 3/2 (5/5) | ✅ | ✅ | ✅ | Correctly conveys that sunlight is multi-colored and blue scatters more off air molecules, delivering the core Rayleigh scattering mechanism in child-friendly terms. |
| How do plants make their own food? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Accurately describes photosynthesis with the full cause-and-effect chain—chlorophyll captures sunlight, water and CO2 combine using that energy to make sugar, releasing oxygen—in child-friendly terms. |
| Why do we have day and night? | 2/2 | 3/2 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation accurately and clearly describes the Earth's rotation relative to the Sun, perfectly conveying the core mechanism of the day/night cycle for a young child. |
| What makes ice melt? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Accurately conveys that heat energizes water molecules causing them to move faster and break their bonds, transitioning ice to liquid water in child-friendly terms. |
| How do magnets work? | 2/2 | 2/2 | — | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: Magnets are special pieces of metal that can push or pull other metal things Correction: Magnets don't have to be metal (e.g., ceramic/ferrite magnets), and they only attract certain metals like iron, nickel, and… |
| Why do things fall to the ground? | 3/2 | 2/2 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation clearly and accurately conveys the core mechanism of gravity as a pulling force exerted by massive objects like the Earth, using perfectly age-appropriate concepts. |
| Where does a puddle go when it dries up? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains evaporation via sun's heat energizing water molecules into vapor, with accurate follow-through to condensation and rain, at an age-appropriate level. |
| Why do we have seasons? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly identifies axial tilt and the angle-of-sunlight mechanism, and notes hemispheres experience opposite seasons. |
| How do our lungs help us breathe? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Accurately describes lungs inflating with air, oxygen transfer to blood, and carbon dioxide removal, delivering the core gas-exchange mechanism at a child-appropriate level. |
| What makes a rainbow? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly conveys refraction, internal reflection, and dispersion in child-friendly terms, with accurate color order and viewing geometry. |
| Why does the moon look like it changes shape? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains that the moon's apparent shape change results from viewing different portions of its sunlit half as it orbits Earth, with an accessible analogy. |
| How do fish breathe underwater? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Accurately describes gills extracting oxygen from water into blood vessels, clearly conveying the core mechanism at a child-friendly level. |

#### Gemini (browser)

| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Why is the sky blue? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly conveys that sunlight contains all colors and that blue light, with shorter waves, scatters more off air molecules, spreading blue across the sky. |
| How do plants make their own food? | 2/2 | 2/2 | — | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: plants use the warm sunlight to mix the water and the air together Correction: It's the light energy (not warmth) that powers photosynthesis; sunlight's heat is not what drives the reaction. |
| Why do we have day and night? | 3/2 | 3/2 | — | 3/2 (5/5) | ✅ | ✅ | ✅ | Correctly explains that Earth's rotation causes day and night by turning locations toward or away from the Sun, in child-friendly language. |
| What makes ice melt? | 3/2 | 3/2 | — | 3/2 (5/5) | ✅ | ✅ | ✅ | Correctly conveys that heat energy increases molecular motion until particles break free from their rigid arrangement, causing melting, in child-appropriate language. |
| How do magnets work? | 2/2 | 2/2 | — | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: electrons spin around like tiny tops Correction: Electron 'spin' is a quantum property, not literal spinning, but the analogy is acceptable for a child; the magnetic effect comes from this spin and from electrons… |
| Why do things fall to the ground? | 2/2 | 3/2 | 2/2 | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: Because the Earth is completely huge, its heavy weight pulls everything straight toward its center. Correction: Gravity is caused by an object's mass (how much stuff it is made of), not its weight (which is the r… |
| Where does a puddle go when it dries up? | 2/2 | 2/2 | — | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: the warm sun and the air shake the tiny pieces of water until they break apart and get very light Correction: Heat gives water molecules more energy so the fastest ones escape into the air as vapor; the molecules… |
| Why do we have seasons? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly identifies axial tilt and explains that direct vs. slanted sunlight concentrates or spreads energy, giving a complete causal chain suitable for a child. |
| How do our lungs help us breathe? | 2/2 | 3/2 | 3/2 | 3/2 (5/5) | ✅ | ✅ | ✅ | The explanation accurately and simply describes the mechanics of breathing and gas exchange without any factual errors. |
| What makes a rainbow? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains refraction, dispersion, and internal reflection in child-friendly terms without factual errors. |
| Why does the moon look like it changes shape? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains that the sun lights the moon and we see varying portions of the lit side as the moon orbits Earth, delivering the core mechanism at a child-appropriate level. |
| How do fish breathe underwater? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Accurately conveys that fish extract dissolved oxygen from water via gills, which transfer it into the blood, in child-friendly language. |

#### Qwen3-4B (base)

| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Why is the sky blue? | 2/2 | 2/2 | — | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: Our eyes see blue light best. Correction: Human eyes are actually most sensitive to green/yellow light, not blue. This claim is inaccurate but not central to the scattering mechanism. |
| How do plants make their own food? | 3/2 | 3/1 | 3/1 | 3/1 (4/5) | ❌ | ❌ | ❌ | The explanation correctly identifies the key ingredients for photosynthesis but mostly lists them without explaining how the energy from the sun is used to change the water and air into food. |
| Why do we have day and night? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains that Earth's rotation causes different sides to face the Sun, producing day and night, with a clear cause-and-effect chain suitable for a child. |
| What makes ice melt? | 3/2 | 2/1 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation correctly identifies that heat (from the sun or a hand) raises the temperature of the ice, causing the phase change from solid to liquid in a clear and age-appropriate manner. |
| How do magnets work? | 2/1 | 1/0 | 2/1 | 2/1 (3/5) | ❌ | ❌ | ❌ | minor: Magnets have special powers... Magnets can make metal things move without touching them. Correction: Magnets create invisible magnetic fields rather than having 'special powers', and they only attract specific me… |
| Why do things fall to the ground? | 3/2 | 3/1 | 3/2 | 3/2 (5/5) | ✅ | ✅ | ✅ | The explanation is highly accurate and clearly conveys the core mechanism of gravity as a pulling force caused by Earth's large mass in a perfectly child-friendly way. |
| Where does a puddle go when it dries up? | 2/2 | 1/1 | 1/1 | 1/1 (2/5) | ❌ | ❌ | ❌ | major: Evaporation means the water turns into air. Correction: Water turns into water vapor (an invisible gas) that mixes with the air; it does not become air. |
| Why do we have seasons? | 2/2 | 2/1 | 2/2 | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: As it does, it tilts on its axis. Correction: Earth's axis maintains a constant tilt in the same direction as it orbits; it does not actively change its tilt or wobble back and forth. |
| How do our lungs help us breathe? | 3/2 | 3/1 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation accurately and clearly describes the process of respiration and gas exchange at a perfect level for a 7-year-old, without any factual errors. |
| What makes a rainbow? | 3/2 | 2/2 | 2/2 | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: The raindrops catch the light and send it back to the sky. Correction: The raindrops reflect the light back toward the observer's eyes so they can see it, rather than just back into the sky. |
| Why does the moon look like it changes shape? | 2/2 | 2/2 | — | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: it is not a solid ball Correction: The moon actually is a solid rocky ball; the phrasing is misleading, though the rest of the explanation correctly describes it as a round rock. |
| How do fish breathe underwater? | 2/2 | 2/1 | 3/2 | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: "They get rid of the extra water and carbon dioxide." Correction: Fish do not mainly get rid of 'extra water' as part of breathing; water passes over the gills, oxygen is taken from it, and carbon dioxide is rele… |

#### Qwen3-0.6B (reference)

| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Why is the sky blue? | 2/1 | 1/1 | 1/0 | 1/1 (2/5) | ❌ | ❌ | ❌ | major: These gases mix together in the atmosphere, and they create a blue color when light from the sun passes through them. Correction: The gases don't create or have a blue color; sunlight contains all colors, and the… |
| How do plants make their own food? | 3/2 | 3/1 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation accurately identifies sunlight, water, and carbon dioxide as the necessary components for photosynthesis, providing a clear and age-appropriate causal mechanism for how plants make food. |
| Why do we have day and night? | 3/2 | 2/1 | 3/1 | 3/1 (4/5) | ❌ | ❌ | ❌ | The explanation correctly identifies Earth's rotation as the cause, but it completely omits the core mechanism of one side of the Earth facing the sun to receive light while the other faces away in darkness. |
| What makes ice melt? | 0/0 | 0/0 | — | 0/0 (0/5) | ❌ | ❌ | ❌ | major: the water in the ice turns into water vapor and then back into liquid water Correction: Melting is a direct change from solid to liquid; ice does not become water vapor first. Ice melts because heat energy makes … |
| How do magnets work? | 1/1 | 0/0 | 0/0 | 0/0 (0/5) | ✅ | ❌ | ❌ | major: Magnetism is like a special kind of 'iron' Correction: Magnetism is an invisible pulling or pushing force, not a physical material like iron. |
| Why do things fall to the ground? | 2/1 | 1/1 | 1/1 | 1/1 (2/5) | ✅ | ❌ | ❌ | major: It doesn't just make them fall—it keeps them in the air until they hit the ground. Correction: Gravity pulls objects down toward the ground; it is not what keeps them in the air. |
| Where does a puddle go when it dries up? | 2/2 | 1/1 | 1/0 | 1/1 (2/5) | ❌ | ❌ | ❌ | major: The water turns back into air Correction: Water does not turn into air; it turns into water vapor, an invisible gas that mixes with the air. |
| Why do we have seasons? | 2/2 | 2/1 | 1/1 | 2/1 (3/5) | ❌ | ❌ | ❌ | minor: Earth's axis is tilted relative to the Sun Correction: The axis is tilted relative to Earth's orbital plane, and this tilt stays pointed in the same direction in space as Earth orbits the Sun. |
| How do our lungs help us breathe? | 2/1 | 2/1 | — | 2/1 (3/5) | ❌ | ❌ | ❌ | minor: air moves in through the nose or mouth, and then through the bronchioles into the lungs Correction: Air actually passes through the trachea and larger bronchi before reaching the smaller bronchioles, and gas exch… |
| What makes a rainbow? | 2/1 | 2/1 | — | 2/1 (3/5) | ❌ | ❌ | ❌ | minor: The sunlight bends and changes direction as it hits the water droplets, creating a colorful pattern. Correction: Light bends when entering and leaving the droplet, and different colors bend by different amounts, … |
| Why does the moon look like it changes shape? | 0/0 | 0/0 | — | 0/0 (0/5) | ❌ | ❌ | ❌ | major: The moon looks like it changes shape because it's actually rotating on its own. Correction: Moon phases are caused by the changing angle between the Sun, Moon, and Earth as the Moon orbits Earth; we see different… |
| How do fish breathe underwater? | 1/1 | 1/1 | — | 1/1 (2/5) | ❌ | ❌ | ❌ | major: gills help them exchange oxygen from the water with the air Correction: Gills extract dissolved oxygen from the water itself; there is no exchange with air. Blood in the gills absorbs oxygen from water and releas… |

#### Qwen3-4B + v2 tune (v4r2)

| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Why is the sky blue? | 2/2 | 2/2 | — | 2/2 (4/5) | ❌ | ✅ | ❌ | minor: Your eyes are very sensitive to blue, so they see that blue light most clearly. Correction: Human eyes are actually most sensitive to green light, not blue. The sky appears blue mainly because blue light scatters… |
| How do plants make their own food? | 3/2 | 3/2 | — | 3/2 (5/5) | ✅ | ✅ | ✅ | Accurately identifies inputs (sunlight, water, CO2), the role of chlorophyll, and the output (sugar) with a clear cause-and-effect chain suitable for a child. |
| Why do we have day and night? | 3/2 | 3/2 | — | 3/2 (5/5) | ✅ | ✅ | ✅ | Correctly explains that Earth's rotation on its axis causes day and night by alternately facing parts of Earth toward and away from the sun over 24 hours. |
| What makes ice melt? | 3/2 | 3/2 | — | 3/2 (5/5) | ✅ | ✅ | ✅ | Correctly identifies heat as the cause and explains that it breaks the bonds holding water molecules in place, allowing them to flow as liquid. |
| How do magnets work? | 3/2 | 2/1 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation is factually accurate and successfully provides the core mechanism of how permanent magnets work (alignment of domains) in an easily understandable way for a child. |
| Why do things fall to the ground? | 2/2 | 2/1 | 3/1 | 2/1 (3/5) | ✅ | ❌ | ❌ | minor: gravity ... is the same everywhere on Earth Correction: Gravity's strength varies slightly by location (e.g., altitude and latitude), though it's roughly similar. |
| Where does a puddle go when it dries up? | 1/1 | 1/1 | — | 1/1 (2/5) | ❌ | ❌ | ❌ | major: it changes into a thin layer of dry soil on the ground Correction: The water does not change into soil; it evaporates into the air as invisible water vapor and/or soaks into the ground, while the soil itself was … |
| Why do we have seasons? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly identifies axial tilt as the cause and clearly explains how tilt changes sun angle and day length to produce seasons. |
| How do our lungs help us breathe? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation correctly describes airflow, gas exchange in alveoli, and the diaphragm's role, delivering the core cause-and-effect chain accessibly. |
| What makes a rainbow? | 2/2 | 2/2 | — | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: First, the sunlight spreads out into many colors... Then the drops bend the light Correction: The order is reversed: the drop bends (refracts) the light first, and that bending is what separates the colors; the s… |
| Why does the moon look like it changes shape? | 1/1 | 0/0 | 0/0 | 0/0 (0/5) | ❌ | ❌ | ❌ | major: As Earth turns, different parts of the moon face the Sun at different times. Correction: Earth's rotation does not cause the moon's phases; the moon orbiting around the Earth changes our viewing angle of its sunl… |
| How do fish breathe underwater? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly explains that gills extract dissolved oxygen from water and exchange it with carbon dioxide via blood capillaries, giving a clear cause-and-effect chain suitable for a child. |

#### Qwen3-4B + v3 tune (v4r3)

| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Why is the sky blue? | 1/1 | 1/1 | — | 1/1 (2/5) | ✅ | ❌ | ❌ | major: Earth blocks most of them Correction: Earth does not block most colors of sunlight; all colors reach the atmosphere, but shorter wavelengths (blue) are scattered more by air molecules than longer wavelengths. |
| How do plants make their own food? | 2/2 | 1/2 | 2/2 | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: Green leaves catch sunlight with a yellow color called chlorophyll. Correction: Chlorophyll is a green pigment, not yellow. |
| Why do we have day and night? | 2/2 | 2/2 | — | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: the side facing the Sun gets closer, and the dark side moves away Correction: The Earth's distance from the Sun doesn't meaningfully change as it rotates; the sides simply rotate to face toward or away from the S… |
| What makes ice melt? | 3/2 | 3/2 | — | 3/2 (5/5) | ✅ | ✅ | ✅ | Correctly explains that heat energy makes water molecules vibrate and break free from their solid structure, becoming liquid. |
| How do magnets work? | 2/1 | 1/1 | 3/2 | 2/1 (3/5) | ❌ | ❌ | ❌ | minor: A magnet is a solid made of a special kind of metal. Correction: Many magnets are made from certain metals or metal-containing materials, but not all magnets are simply a solid metal. |
| Why do things fall to the ground? | 3/2 | 2/2 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation accurately and simply describes gravity as an attractive force exerted by the massive Earth, providing a clear and age-appropriate cause-and-effect mechanism. |
| Where does a puddle go when it dries up? | 2/2 | 2/2 | — | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: The tiny bits of water spread out into the air as a clear smell. Correction: Water vapor is not a smell; it is an invisible gas that spreads into the air. |
| Why do we have seasons? | 2/2 | 1/1 | 1/1 | 1/1 (2/5) | ✅ | ❌ | ❌ | minor: Earth is not a perfect sphere, so it is tilted at a slight angle. Correction: Earth's axial tilt is not caused by its slightly non-spherical shape; it is believed to be the result of ancient collisions. |
| How do our lungs help us breathe? | 3/2 | 2/1 | 3/2 | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation accurately and clearly describes the respiratory process and gas exchange mechanism at an appropriate level for a child. |
| What makes a rainbow? | 3/2 | 3/2 | — | 3/2 (5/5) | ✅ | ✅ | ✅ | Correctly describes refraction, dispersion, and internal reflection in raindrops in child-friendly terms without factual errors. |
| Why does the moon look like it changes shape? | 1/1 | 0/0 | 1/0 | 1/0 (1/5) | ❌ | ❌ | ❌ | major: As Earth spins, your view of the moon changes. Sometimes the whole bright side faces Earth, and sometimes only a sliver does. Correction: The moon's phases are caused by the moon slowly orbiting around the Earth … |
| How do fish breathe underwater? | 1/1 | 0/1 | 1/1 | 1/1 (2/5) | ✅ | ❌ | ❌ | major: Fish need to breathe air to get oxygen, but they cannot swim to the top. Correction: Fish extract dissolved oxygen directly from the water, not from the air, and they are physically capable of swimming to the sur… |

#### Qwen3-4B + v4 tune (v4r4)

| Concept | GPT F/M | Claude F/M | Gemini F/M | Consensus | Read | Accuracy-v2 | Overall | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Why is the sky blue? | 2/2 | 2/2 | — | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: That leaves the yellow and orange light to travel farther up. Correction: The remaining longer-wavelength light (red, orange, yellow) continues traveling forward through the atmosphere toward the observer, not 'f… |
| How do plants make their own food? | 2/2 | 1/2 | 2/2 | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: Any extra sugar gets stored as hard, dry seeds for later. Correction: Extra sugar is typically stored as starch in roots, stems, or leaves; seeds are reproductive structures that contain an embryo along with some… |
| Why do we have day and night? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | Correctly identifies Earth's rotation as the cause and clearly conveys that the sun-facing side experiences day while the opposite side experiences night. |
| What makes ice melt? | 1/1 | 0/0 | 1/0 | 1/0 (1/5) | ❌ | ❌ | ❌ | major: When you step on it, your weight pushes hard against the ice. That heavy push gives the parts enough energy to break free from their frozen spots. Now that they are free, the tiny pieces flow together to make liq… |
| How do magnets work? | 2/1 | 1/1 | 1/0 | 1/1 (2/5) | ✅ | ❌ | ❌ | major: When a magnet moves, these bits push on other metals to draw them close. Correction: A magnet does not need to move to attract metals; the aligned magnetic domains create a magnetic field around the magnet that a… |
| Why do things fall to the ground? | 1/1 | 1/1 | — | 1/1 (2/5) | ✅ | ❌ | ❌ | major: If the object is light, gravity simply drags it down slowly. A heavy object falls faster because its weight is much harder to stop. Correction: In the absence of significant air resistance, objects fall at the sa… |
| Where does a puddle go when it dries up? | 2/2 | 2/2 | — | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: This gas floats away into the sky until it meets a cold cloud. The cold cloud turns the gas back into tiny drops of water. Correction: Water vapor rises and cools in the sky, and this cooling causes it to condens… |
| Why do we have seasons? | 1/1 | 0/0 | 0/0 | 0/0 (0/5) | ✅ | ❌ | ❌ | major: When our planet spins around the sun, one side gets closer to the bright sun. That side gets warm and has summer, while the far side gets cold winter. Correction: Seasons are caused by the angle of sunlight (dire… |
| How do our lungs help us breathe? | 3/2 | 2/2 | 3/2 | 3/2 (5/5) | ✅ | ✅ | ✅ | The explanation perfectly captures the core mechanism of gas exchange (oxygen in, carbon dioxide out) with accurate, age-appropriate language and no factual errors. |
| What makes a rainbow? | 2/2 | 2/2 | — | 2/2 (4/5) | ✅ | ✅ | ✅ | minor: Red bends the least, so it rolls to the bottom of the arc. Blue bends a lot, so it climbs up to the top. Correction: In a primary rainbow, red actually appears at the top of the arc and blue/violet at the bottom,… |
| Why does the moon look like it changes shape? | 1/1 | 0/0 | 0/0 | 0/0 (0/5) | ✅ | ❌ | ❌ | major: Earth slowly moves around the dark moon. Correction: The moon orbits around the Earth; the Earth does not orbit the moon. |
| How do fish breathe underwater? | 3/2 | 3/2 | — | 3/2 (5/5) | ❌ | ✅ | ❌ | The explanation correctly describes water flowing over gills where blood absorbs oxygen, conveying the core mechanism in child-friendly terms. |
<!-- accuracy-v2:end -->

---

## Historical Accuracy-v1 Results

The following tables are preserved from the original 0/1/2 accuracy rubric.

<!-- accuracy-v1-historical:start -->
Same 12 concepts, same saved outputs, same accuracy judgments as `results_v3.md` (gpt-4o, audience-calibrated). Only the readability gate changed: v3's FK 1.5-3.0 band was shown to target ~grade 1-2 (`eval/metric_comparison_real.md`); v4 uses the recalibrated real-grade-3 band FK 3-6 plus the co-best co-metric ARI 3-7.

`overall_pass = readability_pass_v4 AND accuracy==2`

## Headline (v4)

| Model | readability (v4) | accuracy=2 | overall pass |
|---|---|---|---|
| GPT (gpt-4o) | 2/12 | 12/12 | **2/12** |
| Claude (browser) | 1/12 | 12/12 | **1/12** |
| Gemini (browser) | 4/12 | 12/12 | **4/12** |
| Qwen3-4B (local, instruct-style) | 2/12 | 12/12 | **2/12** |
| Qwen3-0.6B (local, reference) | 2/12 | 5/12 | **0/12** |

## GPT (gpt-4o)  —  `gpt-4o`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 5.04 | 6.09 | 2.97 | ✅ | ✅ | ❌ | 2 | ❌ |
| How do plants make their own food? | 3.66 | 7.55 | 2.71 | ✅ | ❌ | ❌ | 2 | ❌ |
| Why do we have day and night? | 1.82 | 2.65 | 1.26 | ❌ | ❌ | ❌ | 2 | ❌ |
| What makes ice melt? | 2.64 | 3.28 | 1.6 | ❌ | ✅ | ❌ | 2 | ❌ |
| How do magnets work? | 5.04 | 6.48 | 1.88 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why do things fall to the ground? | 4.69 | 4.49 | 1.74 | ✅ | ✅ | ❌ | 2 | ❌ |
| Where does a puddle go when it dries up? | 3.65 | 2.75 | 1.86 | ✅ | ❌ | ❌ | 2 | ❌ |
| Why do we have seasons? | 2.78 | 3.93 | 1.73 | ❌ | ✅ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 3.84 | 4.39 | 1.08 | ✅ | ✅ | ✅ | 2 | ✅ |
| What makes a rainbow? | 4.45 | 5.52 | 0.99 | ✅ | ✅ | ✅ | 2 | ✅ |
| Why does the moon look like it changes shape? | 1.86 | 2.79 | 2.04 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do fish breathe underwater? | 3.16 | 2.98 | 2.21 | ✅ | ❌ | ❌ | 2 | ❌ |

## Claude (browser)  —  `browser (manual paste)`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 3.65 | 4.0 | 1.52 | ✅ | ✅ | ✅ | 2 | ✅ |
| How do plants make their own food? | 5.77 | 6.44 | 2.36 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why do we have day and night? | 2.38 | 2.34 | 0.36 | ❌ | ❌ | ❌ | 2 | ❌ |
| What makes ice melt? | 3.64 | 3.69 | 1.85 | ✅ | ✅ | ❌ | 2 | ❌ |
| How do magnets work? | 5.83 | 6.31 | 2.41 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why do things fall to the ground? | 6.24 | 6.14 | 3.01 | ❌ | ✅ | ❌ | 2 | ❌ |
| Where does a puddle go when it dries up? | 5.83 | 4.95 | 1.92 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why do we have seasons? | 6.03 | 7.52 | 1.26 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 3.75 | 4.43 | 2.63 | ✅ | ✅ | ❌ | 2 | ❌ |
| What makes a rainbow? | 6.77 | 7.32 | 1.12 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why does the moon look like it changes shape? | 3.98 | 4.46 | 2.21 | ✅ | ✅ | ❌ | 2 | ❌ |
| How do fish breathe underwater? | 4.96 | 6.52 | 3.0 | ✅ | ✅ | ❌ | 2 | ❌ |

## Gemini (browser)  —  `browser (manual paste)`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 5.86 | 7.15 | 2.29 | ✅ | ❌ | ❌ | 2 | ❌ |
| How do plants make their own food? | 4.43 | 5.54 | 1.61 | ✅ | ✅ | ✅ | 2 | ✅ |
| Why do we have day and night? | 3.2 | 4.71 | 1.24 | ✅ | ✅ | ✅ | 2 | ✅ |
| What makes ice melt? | 5.5 | 5.54 | 1.55 | ✅ | ✅ | ✅ | 2 | ✅ |
| How do magnets work? | 6.99 | 7.92 | 1.12 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why do things fall to the ground? | 7.31 | 8.71 | 2.36 | ❌ | ❌ | ❌ | 2 | ❌ |
| Where does a puddle go when it dries up? | 7.01 | 6.94 | 1.2 | ❌ | ✅ | ❌ | 2 | ❌ |
| Why do we have seasons? | 6.22 | 7.4 | 1.64 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 3.4 | 5.24 | 0.98 | ✅ | ✅ | ✅ | 2 | ✅ |
| What makes a rainbow? | 6.18 | 8.04 | 1.91 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why does the moon look like it changes shape? | 4.91 | 5.36 | 2.37 | ✅ | ✅ | ❌ | 2 | ❌ |
| How do fish breathe underwater? | 5.46 | 5.56 | 1.88 | ✅ | ✅ | ❌ | 2 | ❌ |

## Qwen3-4B (local, instruct-style)  —  `Qwen/Qwen3-4B`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 0.56 | 0.45 | 0.95 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do plants make their own food? | 2.75 | 5.64 | 1.34 | ❌ | ✅ | ❌ | 2 | ❌ |
| Why do we have day and night? | 2.36 | 2.94 | 0.73 | ❌ | ❌ | ❌ | 2 | ❌ |
| What makes ice melt? | 1.52 | 1.97 | 1.05 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do magnets work? | 2.55 | 4.39 | 1.3 | ❌ | ✅ | ❌ | 2 | ❌ |
| Why do things fall to the ground? | 3.88 | 5.34 | 1.68 | ✅ | ✅ | ✅ | 2 | ✅ |
| Where does a puddle go when it dries up? | 2.88 | 1.07 | 1.28 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why do we have seasons? | 2.29 | 3.1 | 1.0 | ❌ | ✅ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 3.64 | 2.4 | 2.54 | ✅ | ❌ | ❌ | 2 | ❌ |
| What makes a rainbow? | 3.83 | 4.84 | 1.74 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why does the moon look like it changes shape? | 2.35 | 3.08 | 1.86 | ❌ | ✅ | ❌ | 2 | ❌ |
| How do fish breathe underwater? | 3.57 | 3.22 | 0.9 | ✅ | ✅ | ✅ | 2 | ✅ |

## Qwen3-0.6B (local, reference)  —  `Qwen/Qwen3-0.6B`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 7.25 | 8.38 | 1.98 | ❌ | ❌ | ❌ | 1 | ❌ |
| How do plants make their own food? | 4.96 | 7.14 | 1.96 | ✅ | ❌ | ❌ | 2 | ❌ |
| Why do we have day and night? | 7.35 | 8.43 | 1.25 | ❌ | ❌ | ❌ | 2 | ❌ |
| What makes ice melt? | 8.73 | 9.14 | 3.06 | ❌ | ❌ | ❌ | 0 | ❌ |
| How do magnets work? | 5.86 | 6.94 | 0.41 | ✅ | ✅ | ✅ | 0 | ❌ |
| Why do things fall to the ground? | 5.25 | 6.68 | 1.53 | ✅ | ✅ | ✅ | 1 | ❌ |
| Where does a puddle go when it dries up? | 6.12 | 5.74 | 2.45 | ❌ | ✅ | ❌ | 2 | ❌ |
| Why do we have seasons? | 6.8 | 7.66 | 1.2 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 8.95 | 9.67 | 2.47 | ❌ | ❌ | ❌ | 1 | ❌ |
| What makes a rainbow? | 7.56 | 9.08 | 2.25 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why does the moon look like it changes shape? | 5.35 | 6.43 | 2.49 | ✅ | ✅ | ❌ | 0 | ❌ |
| How do fish breathe underwater? | 7.97 | 8.96 | 1.6 | ❌ | ❌ | ❌ | 0 | ❌ |

## Failure-mode breakdown (why they miss the v4 band)

Accuracy is saturated (all four at 12/12), so **readability is the only
differentiator.** Splitting the readability failures (dispersion cap = 1.7):

| Model | in FK+ARI band | pass v4 (+even) | too simple (<floor) | too hard (>ceiling) | uneven (stdev>1.7) |
|---|---|---|---|---|---|
| GPT | 5/12 | 2/12 | 6 | 1 | 8 |
| Claude | 8/12 | 1/12 | 1 | 3 | 8 |
| Gemini | 6/12 | 4/12 | 0 | 6 | 5 |
| Qwen3-4B | 3/12 | 2/12 | 9 | 0 | 3 |

Reading:
- **Unevenness (dispersion) is still the most common binding constraint** for the
  frontier models (5–8 of 12), even after loosening the cap to 1.7. Passages sit in
  the FK+ARI grade-3 band on average but lurch between a baby-talk sentence and a
  hard one.
- **The models miss in opposite directions.** Qwen3-4B is *too simple* (9/12 below
  the grade-3 floor — terse, sub-grade-3 vocabulary; this is why it looked good under
  the old FK 1.5–3.0 band). Gemini skews *too hard* (6/12 over ceiling). GPT/Claude
  straddle the band but read unevenly.
- **Thesis intact.** Under a band calibrated to *real* grade-3 reading level, no
  prompted model reliably hits grade-3 + accurate + even — best is 4/12 (Gemini). The
  4B fine-tune target sits at 2/12 at baseline. Fine-tuning is still the justified
  move; what changed is the *target* (pull text into FK 3–6 evenly, not down to FK ~2).
<!-- accuracy-v1-historical:end -->
