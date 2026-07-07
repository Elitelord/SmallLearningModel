# Litmus results

Behavior Spec: no sentence > FK 3.0; ≥70% of sentences in FK 2.0–3.0; factually correct AND conveys the mechanism.

`overall_pass = readability_pass AND accuracy_pass (score == 2)`

## Headline

- **GPT (gpt-4o, API)** passed the full spec on **0/12** concepts (readability 0/12, accuracy=2 on 12/12). Dominant failure mode: sentence(s) over FK 3.0 ceiling.
- **Gemini (browser)** passed the full spec on **0/12** concepts (readability 0/12, accuracy=2 on 12/12). Dominant failure mode: sentence(s) over FK 3.0 ceiling.
- **Claude (browser)** passed the full spec on **1/12** concepts (readability 1/12, accuracy=2 on 12/12). Dominant failure mode: sentence(s) over FK 3.0 ceiling.
- **Qwen3-0.6B (local)** passed the full spec on **0/12** concepts (readability 0/12, accuracy=2 on 5/12). Dominant failure mode: sentence(s) over FK 3.0 ceiling.

## GPT (gpt-4o, API)  —  model_id=`gpt-4o`, temp=0.7

| Concept | max_fk | over | %band | read | acc | overall | failure note |
|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 11.06 | 6 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 3.1 (> 3.0 ceiling); 6 over, max 11.06 |
| How do plants make their own food? | 8.76 | 4 | 25% | ❌ | 2 | ❌ | sentence 4 hits FK 4.91 (> 3.0 ceiling); 4 over, max 8.76 |
| Why do we have day and night? | 3.65 | 1 | 50% | ❌ | 2 | ❌ | sentence 8 hits FK 3.65 (> 3.0 ceiling); 1 over, max 3.65 |
| What makes ice melt? | 6.79 | 3 | 43% | ❌ | 2 | ❌ | sentence 3 hits FK 6.79 (> 3.0 ceiling); 3 over, max 6.79 |
| How do magnets work? | 9.08 | 6 | 12% | ❌ | 2 | ❌ | sentence 1 hits FK 4.79 (> 3.0 ceiling); 6 over, max 9.08 |
| Why do things fall to the ground? | 7.63 | 7 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 4.83 (> 3.0 ceiling); 7 over, max 7.63 |
| Where does a puddle go when it dries up? | 9.57 | 6 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 3.72 (> 3.0 ceiling); 6 over, max 9.57 |
| Why do we have seasons? | 7.16 | 3 | 25% | ❌ | 2 | ❌ | sentence 3 hits FK 3.63 (> 3.0 ceiling); 3 over, max 7.16 |
| How do our lungs help us breathe? | 5.99 | 6 | 12% | ❌ | 2 | ❌ | sentence 3 hits FK 4.83 (> 3.0 ceiling); 6 over, max 5.99 |
| What makes a rainbow? | 6.37 | 7 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 4.83 (> 3.0 ceiling); 7 over, max 6.37 |
| Why does the moon look like it changes shape? | 5.04 | 3 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 4.2 (> 3.0 ceiling); 3 over, max 5.04 |
| How do fish breathe underwater? | 6.42 | 4 | 25% | ❌ | 2 | ❌ | sentence 1 hits FK 6.42 (> 3.0 ceiling); 4 over, max 6.42 |

## Gemini (browser)  —  model_id=`browser (manual paste)`, temp=uncontrolled

| Concept | max_fk | over | %band | read | acc | overall | failure note |
|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 9.01 | 5 | 28% | ❌ | 2 | ❌ | sentence 1 hits FK 6.78 (> 3.0 ceiling); 5 over, max 9.01 |
| How do plants make their own food? | 6.14 | 5 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 5.04 (> 3.0 ceiling); 5 over, max 6.14 |
| Why do we have day and night? | 4.83 | 5 | 14% | ❌ | 2 | ❌ | sentence 2 hits FK 4.83 (> 3.0 ceiling); 5 over, max 4.83 |
| What makes ice melt? | 7.61 | 6 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 5.86 (> 3.0 ceiling); 6 over, max 7.61 |
| How do magnets work? | 8.54 | 7 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 8.54 (> 3.0 ceiling); 7 over, max 8.54 |
| Why do things fall to the ground? | 12.3 | 6 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 12.3 (> 3.0 ceiling); 6 over, max 12.3 |
| Where does a puddle go when it dries up? | 9.96 | 6 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 6.01 (> 3.0 ceiling); 6 over, max 9.96 |
| Why do we have seasons? | 9.08 | 6 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 3.89 (> 3.0 ceiling); 6 over, max 9.08 |
| How do our lungs help us breathe? | 4.91 | 3 | 50% | ❌ | 2 | ❌ | sentence 1 hits FK 3.72 (> 3.0 ceiling); 3 over, max 4.91 |
| What makes a rainbow? | 9.14 | 6 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 6.01 (> 3.0 ceiling); 6 over, max 9.14 |
| Why does the moon look like it changes shape? | 7.7 | 4 | 17% | ❌ | 2 | ❌ | sentence 1 hits FK 5.4 (> 3.0 ceiling); 4 over, max 7.7 |
| How do fish breathe underwater? | 8.9 | 6 | 14% | ❌ | 2 | ❌ | sentence 2 hits FK 6.31 (> 3.0 ceiling); 6 over, max 8.9 |

## Claude (browser)  —  model_id=`browser (manual paste)`, temp=uncontrolled

| Concept | max_fk | over | %band | read | acc | overall | failure note |
|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 5.81 | 3 | 33% | ❌ | 2 | ❌ | sentence 2 hits FK 4.96 (> 3.0 ceiling); 3 over, max 5.81 |
| How do plants make their own food? | 9.26 | 6 | 14% | ❌ | 2 | ❌ | sentence 1 hits FK 3.65 (> 3.0 ceiling); 6 over, max 9.26 |
| Why do we have day and night? | 2.85 | 0 | 80% | ✅ | 2 | ✅ | pass |
| What makes ice melt? | 6.28 | 5 | 0% | ❌ | 2 | ❌ | sentence 2 hits FK 6.28 (> 3.0 ceiling); 5 over, max 6.28 |
| How do magnets work? | 8.76 | 6 | 14% | ❌ | 2 | ❌ | sentence 1 hits FK 8.39 (> 3.0 ceiling); 6 over, max 8.76 |
| Why do things fall to the ground? | 11.77 | 5 | 17% | ❌ | 2 | ❌ | sentence 1 hits FK 4.82 (> 3.0 ceiling); 5 over, max 11.77 |
| Where does a puddle go when it dries up? | 8.54 | 6 | 14% | ❌ | 2 | ❌ | sentence 1 hits FK 8.54 (> 3.0 ceiling); 6 over, max 8.54 |
| Why do we have seasons? | 7.85 | 5 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 5.4 (> 3.0 ceiling); 5 over, max 7.85 |
| How do our lungs help us breathe? | 8.59 | 3 | 14% | ❌ | 2 | ❌ | sentence 4 hits FK 5.88 (> 3.0 ceiling); 3 over, max 8.59 |
| What makes a rainbow? | 9.32 | 6 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 6.71 (> 3.0 ceiling); 6 over, max 9.32 |
| Why does the moon look like it changes shape? | 6.73 | 4 | 17% | ❌ | 2 | ❌ | sentence 1 hits FK 4.0 (> 3.0 ceiling); 4 over, max 6.73 |
| How do fish breathe underwater? | 9.21 | 4 | 28% | ❌ | 2 | ❌ | sentence 1 hits FK 7.59 (> 3.0 ceiling); 4 over, max 9.21 |

## Qwen3-0.6B (local)  —  model_id=`Qwen/Qwen3-0.6B`, temp=0.7

| Concept | max_fk | over | %band | read | acc | overall | failure note |
|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 8.9 | 4 | 0% | ❌ | 0 | ❌ | sentence 1 hits FK 7.61 (> 3.0 ceiling); 4 over, max 8.9 |
| How do plants make their own food? | 8.38 | 3 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 3.65 (> 3.0 ceiling); 3 over, max 8.38 |
| Why do we have day and night? | 9.08 | 3 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 9.08 (> 3.0 ceiling); 3 over, max 9.08 |
| What makes ice melt? | 12.36 | 3 | 0% | ❌ | 0 | ❌ | sentence 1 hits FK 8.47 (> 3.0 ceiling); 3 over, max 12.36 |
| How do magnets work? | 6.73 | 4 | 0% | ❌ | 0 | ❌ | sentence 1 hits FK 6.73 (> 3.0 ceiling); 4 over, max 6.73 |
| Why do things fall to the ground? | 7.77 | 4 | 0% | ❌ | 1 | ❌ | sentence 1 hits FK 3.65 (> 3.0 ceiling); 4 over, max 7.77 |
| Where does a puddle go when it dries up? | 8.76 | 2 | 33% | ❌ | 1 | ❌ | sentence 2 hits FK 8.76 (> 3.0 ceiling); 2 over, max 8.76 |
| Why do we have seasons? | 8.59 | 4 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 6.79 (> 3.0 ceiling); 4 over, max 8.59 |
| How do our lungs help us breathe? | 12.86 | 4 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 12.86 (> 3.0 ceiling); 4 over, max 12.86 |
| What makes a rainbow? | 10.56 | 4 | 0% | ❌ | 2 | ❌ | sentence 1 hits FK 5.82 (> 3.0 ceiling); 4 over, max 10.56 |
| Why does the moon look like it changes shape? | 8.59 | 3 | 40% | ❌ | 0 | ❌ | sentence 1 hits FK 7.57 (> 3.0 ceiling); 3 over, max 8.59 |
| How do fish breathe underwater? | 11.47 | 4 | 0% | ❌ | 0 | ❌ | sentence 1 hits FK 5.68 (> 3.0 ceiling); 4 over, max 11.47 |
