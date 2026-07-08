# Grade-3 Science Reading-Material Source Survey

Time-boxed survey of external reading material that could feed a data-generation
**rewrite loop** for the grade-3 science fine-tuning project. Use is
non-commercial research, so licensing latitude is wide, but every source's
license is recorded below.

**Scoring:** each sampled text unit was run through the repo's operative
readability gate (`litmus/fk_score.py`, `readability_pass_v3`). Pass requires
whole-passage FK in [1.5, 3.0] **and** per-sentence FK std-dev ≤ 1.3 (over ≥8-word
sentences) **and** no ≥10-word sentence over FK 4.0. "Pass-rate" = % of the
sample with `readability_pass_v3 == True`.

## Headline finding

**Nothing passes the v3 gate as drop-in data.** Every genuinely samplable source
scored 0–5%. The gate targets a very tight grade-3 band (whole-passage FK 1.5–3.0
with low dispersion); real published reading material — even "simplified" material
— sits at grade ~5–12. So the useful role of any external source here is as a
**seed** for the rewrite loop, not as reference explanations. Two axes of value:

- **Topic / prompt seeds** — what to write about (concept lists, exam questions).
- **Style seeds** — plain-language phrasing to imitate/simplify.

## Ranked table

| # | Source | License | Format | Size | v3 pass-rate (sample) | Recommendation |
|---|--------|---------|--------|------|-----------------------|----------------|
| 1 | **Simple English Wikipedia** | CC BY-SA 4.0 | Raw encyclopedic prose (needs restructuring) | ~250k articles | **0% (0/12)** | **Fold in as STYLE + TOPIC seed.** Clean, science-rich, permissive. Grade ~5–8, so every unit must go through the rewrite loop. |
| 2 | **ARC-Easy** (allenai/ai2_arc) | CC BY-SA 4.0 | Grade-school science multiple-choice questions | ~5,200 Q (Easy) | N/A (prompt source, not prose) | **Fold in as PROMPT/TOPIC seed.** Genuine grade-school science stems; clean; feed as concept prompts to the loop. Some vocab (meiosis, metamorphic) skews grade 6-8. |
| 3 | SciQ (allenai/sciq) | CC BY-NC 3.0 | Science Q + support paragraph | 13,679 Q | not sampled (support text is Wikipedia-sourced, grade 5+) | Optional prompt seed. Questions usable as topics; support paragraphs are complex, not drop-in. |
| 4 | CK-12 elementary science FlexBooks | CC BY-NC | Lesson prose (needs restructuring) | Grades 1–5 band | not sampled (site returns 403 to fetch) | Weak maybe. Good license + on-topic, but reading level ~grade 4–6 and not scrapable without effort; would fail gate like Simple Wiki. |
| 5 | CommonLit CLEAR Corpus | Mostly public domain; some CC BY / CC BY-SA 3.0 | Graded reading excerpts (literary + informational) | ~5,000 excerpts | **0% (0/20)** | Skip as data. Authentic 250-yr prose, grade 3–12, high dispersion. Topic seed at best. |
| 6 | ASSET (facebook/asset) | CC BY-SA 4.0 | Sentence-level simplification pairs | 2,359 × 10 refs | **5% (1/20)** | Skip. Generic Wikipedia domain (not science), single sentences, grade 5–9. |
| 7 | WikiLarge / WikiSmall | CC BY-SA 4.0 | Aligned complex→simple sentence pairs | 296k / 88k pairs | ~0% (not directly sampled; Simple-Wiki-derived, same register) | Skip. Same style ceiling as Simple Wiki, plus noisy automatic alignments. |
| 8 | ELI5 (r/explainlikeimfive) | Reddit content; HF "eli5" defunct/removed over licensing | Long-form Q→A | ~270k QA | not sampled (adult register) | Skip. "Explain simply" but written for adults (~grade 9+); licensing murky. |
| 9 | CommonLit.org site materials | CC BY-NC-SA 4.0 | Leveled ELA passages + questions | grade 3–12 | not sampled (site, not bulk-downloadable) | Skip as data; grade 3 band is a small slice and still authentic prose. |
| 10 | **ReadWorks** | Proprietary — "may not post on an open platform"; license required for reuse | Leveled passages | large | not sampled | **Skip on licensing.** Not an open license even for non-commercial redistribution. |
| 11 | **Newsela** | Restricted — contract/NDA only, no redistribution | Professional multi-level rewrites | large | not sampled | **Skip.** Highest-quality leveled rewrites but access-gated; can't obtain or redistribute. |

## Per-source notes (sampled evidence)

### Simple English Wikipedia — 0/12 (0%)
12 science article intros (Photosynthesis, Water cycle, Volcano, Frog, Sun, Rain,
Magnet, Butterfly, Spider, Electricity, Dinosaur, Moon). Whole-passage FK ranged
**4.4–12.3** (median ~8), well above the 3.0 ceiling; most also carried ≥1 long
sentence over FK 4.0. Even the "simple" encyclopedia leans on taxonomy and jargon
("air-breathing arthropods", "convergent evolution", "convert light energy into
chemical energy"). Confirms the brief's expectation: **style/topic seed, not
drop-in.** Its value is topical breadth + a permissive license, and the phrasing
is close enough that the rewrite loop has an easy job.

### ARC-Easy — prompt source (pass-rate N/A)
Sampled 15 question stems: all clean, on-topic grade-school science
("Which two body systems are directly involved in movement?", "Which change in
the state of water particles causes the particles to become arranged in a fixed
position?"). These are exactly the kind of concept prompts the rewrite loop needs
to be pointed at. Not explanation prose, so the readability gate does not apply to
the source itself — it applies to what the loop generates from it.

### CommonLit CLEAR Corpus — 0/20 (0%)
20 excerpts spread across the corpus. Whole-passage FK **5.6–19.1**, dispersion
std-dev **2–6** (far over the 1.3 cap) — authentic literary + informational prose
with long, uneven sentences. Open license is attractive but the text is nowhere
near the target band.

### ASSET — 1/20 (5%)
20 simplified sentences from the validation split. Domain is generic Wikipedia
(kiwi fruit, WWE, prisons, WWII delegates), **not science**. As single sentences
they mostly land grade 5–13; the one pass was a short two-sentence date/location
fragment. Wrong domain and wrong granularity for this project.

## Bottom line

Fold in **Simple English Wikipedia (CC BY-SA 4.0)** as the style + topic seed and
**ARC-Easy (CC BY-SA 4.0)** as the prompt/topic seed. Neither is usable as-is
(Simple Wiki scored 0/12 on the v3 gate; ARC-Easy is questions, not prose), which
is expected — the whole point of the project's rewrite loop is to take clean,
on-topic, permissively-licensed source material at grade ~5–8 and drive it down
into the grade-3 band. Simple Wiki supplies plain-language science phrasing close
enough that the loop's job is light; ARC-Easy supplies a few thousand genuine
grade-school science concepts to aim the loop at. Everything else is either the
wrong domain (ASSET/WikiLarge), the wrong reading level with no licensing upside
(CLEAR, ELI5), or licensing-blocked (ReadWorks, Newsela). No source should be
integrated as reference explanations — Sameer decides what, if anything, to fold in.
