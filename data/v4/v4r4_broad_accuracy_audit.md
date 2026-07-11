# v4r4 broad accuracy audit

The deterministic audit sampled 40 training targets across distinct concepts without
using litmus failure families.

| Metric | Result |
|---|---:|
| Readability pass | 40/40 |
| Clean accuracy (F3/M2) | 30/40 |
| Tolerant accuracy-v2 | 39/40 |
| Overall-v2 | 39/40 |
| Primary-judge disagreements | 9/40 |

The single tolerant failure taught the wrong mechanism for tree height: it said wood
layers under the bark stack upward, while those layers mainly increase trunk width;
height comes from growth at shoot tips. The other nine records contain localized
imprecision and pass tolerant accuracy-v2.

Decision: do not replay v4r4 wholesale. Export only the 30 consensus-clean audited
records, then generate the rest of v4r5 with the clean accuracy-v2 gate. This preserves
a small broad replay component without admitting known minor or major errors.
