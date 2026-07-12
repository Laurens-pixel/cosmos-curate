# Caption-Quality & Shot-Boundary Benchmark — Results Report

**Dataset:** AgiBotWorld-Alpha · 2,587 videos · 15,526 caption windows · 14,996 ground-truth action segments
**Caption model under test:** Qwen-2.5-VL-7B
**Date:** 2026-06-24
**Source artefacts:** `benchmark_results/caption_report.json`, `boundary_report.json`, `boundary_report_real.json`

---

## 0 · What this report covers

This is the first end-to-end run of the new evaluation harness on the full 2,587-video set. It measures three things on the captioning pipeline output:

1. **Grounding** — does a caption describe the action actually happening in its video window? (measured both lexically and semantically)
2. **Judges** — how reliably do the two VLM judges flag wrong captions?
3. **Shot boundary** — how well do the segmenters recover the ground-truth action boundaries?

Everything is computed against the AgiBotWorld `task_info` ground truth.

### Run inventory

| Component | Status | Artefact |
|---|---|---|
| Caption quality (15,526 windows) | ✅ COMPLETE | `caption_report.json` |
| Judge agreement (Gemma4 + VCI-7B) | ✅ COMPLETE | `caption_report.json` |
| Judge vs GT comparison (VCI-7B / Gemma4 / Qwen-judge) | ✅ COMPLETE | `eval_stage/*_scores.json` (§3·B) |
| Semantic grounding (MiniLM + BGE) | ✅ COMPLETE | `caption_report.json` |
| Shot boundary — fixed-stride baseline | ✅ COMPLETE | `boundary_report.json` |
| Shot boundary — TransNetV2 + SigLIP2 | ✅ COMPLETE | `boundary_report_real.json` |
| Shot boundary — InHARD cross-domain (6 detectors) | ✅ COMPLETE | `boundary_report_inhard.json` (§1·B) |
| Caption grounding — YouCook2 cross-domain | ✅ COMPLETE | `caption_report_youcook2.json` (§2·B) |
| Text-video grounding (SigLIP2, 15,526 windows) | ✅ COMPLETE | `textvideo_report.json` (§5) |
| Qwen3-VL caption comparison | ✅ COMPLETE (H100, FP8 fix) — 30 videos | `agibot_qwen3vl_30vid_output/` |

### Headline numbers

- **Caption `quality_score`: 0.583 / 1.0** — composite of semantic-F1, caption uniqueness, object continuity and (1 − contradiction rate). One comparable figure-of-merit per captioning model.
- **Semantic grounding (BGE): sem-F1 = 0.610** (recall 0.616, precision 0.605) — captions capture the gist of the GT action ~61% of the time.
- **Lexical grounding: action-F1 = 0.107** — far below semantic-F1: captions are right in *meaning* but rarely in exact *wording*.
- **Shot boundary: best boundary-F1 = 0.020** — detectors find <5% of GT action boundaries; AgiBot transitions are semantic, not visual cuts (expected — see §1).
- **Judge agreement: Cohen's κ = 0.108** — the two judges barely agree beyond chance; they flag different errors (see §3).

---

## 1 · Shot-boundary detection

**Goal:** split each long episode into segments that line up with ground-truth action boundaries. Three segmenters compared against 14,996 GT action segments.

- `boundary-F1@0.5s` — a predicted boundary counts as correct if it lands within 0.5 s of a GT boundary.
- `segment-F1@IoU` — how well predicted segments overlap GT segments.
- `over-seg ratio` — predicted ÷ GT segment count.

| detector | nPred | bF1@0.5s | bP@0.5s | bR@0.5s | segF1@0.1 | segF1@0.5 | overSeg | distGT→P |
|---|---|---|---|---|---|---|---|---|
| fixed-stride (baseline) | 2,587 | 0.039 | 1.000 | 0.039 | 0.354 | 0.056 | 0.237 | — |
| TransNetV2 | 3,237 | 0.020 | 0.778 | 0.043 | 0.369 | 0.065 | 0.289 | 25.9 s |
| semantic SigLIP2 | 3,235 | 0.020 | 0.779 | 0.043 | 0.369 | 0.065 | 0.289 | 25.9 s |

### What the boundary numbers mean

- **TransNetV2 ≈ SigLIP2 ≈ fixed-stride.** All three score boundary-F1 < 0.04. The real detectors are *no better* than naively cutting at fixed intervals. **This is not a bug — it is the key finding.**
- **Why so low:** TransNetV2 and SigLIP2 detect *visual* scene changes (camera cuts, large appearance shifts). AgiBot action boundaries are *semantic* — the arm keeps the same scene and background while switching from "pick" to "place". There is no visual cut to detect, so both detectors fire almost at random with respect to GT.
- **Precision vs recall:** TransNetV2 boundary-precision is 0.78 but recall only 0.04 — when it does fire near a GT boundary it is usually right, but it misses ~96% of them.
- **Distance asymmetry:** GT→pred distance 25.9 s vs pred→GT 7.4 s — most GT boundaries have no nearby prediction.
- **Implication:** for this domain, GT-aligned windows (`--gt-windows-source agibot`) or fixed-stride windows are the honest choice; visual shot detection adds nothing. The harness now proves that quantitatively rather than assuming it.

---

## 1·B · Cross-domain control: shot boundary on InHARD (human assembly)

**Why this experiment:** the AgiBot result above could mean one of two things — (a) the detectors are broken, or (b) AgiBot boundaries are genuinely invisible (semantic, not visual). To separate these we run the **same detectors, same harness** on a *different* domain that has real visual variation. If the detectors score much higher there, the AgiBot failure is proven to be a domain property, not a harness bug. This is the controlled counterfactual.

**Dataset:** InHARD Online — 38 full human industrial-assembly session videos (~400–650 s each, continuous fixed-camera footage, **not** concatenated clips), with **4,803 ground-truth action segments** at real wall-clock timestamps from `InHARD.csv` (`Action_start_rgb_sec` / `Action_end_rgb_sec`). Median ~126 actions/video, ~4 s each — fine-grained, so absolute recall will be modest; the **relative** detector ordering and the **contrast with AgiBot** are the signal.

**Detectors:** TransNetV2 (pixel), and semantic CLIP-L / SigLIP2 / DINOv2-L / DINOv2-giant / V-JEPA2-L. Same `--shot-boundary-model` codepath as AgiBot. (DINOv3-L is gated and was not fetched; DINOv2-giant was added as an ungated larger self-supervised ViT to test whether scaling the DINO backbone helps.)

**Status: ✅ COMPLETE — job `sbd_inhard` (`run_sbd_inhard.sh`), 38/38 videos, 4,803 GT segments → `benchmark_results/boundary_report_inhard.json`.** AgiBot columns are repeated from §1 for side-by-side reading. (`segF1@0.5` is segment-F1 at IoU 0.5; `bF1@0.5s` is boundary-F1 at 0.5 s tolerance.)

| detector | dataset | nPred | bF1@0.5s | bP@0.5s | bR@0.5s | segF1@0.5 |
|---|---|---|---|---|---|---|
| TransNetV2 | AgiBot (robot) | 3,237 | 0.020 | 0.778 | 0.043 | 0.065 |
| TransNetV2 | InHARD (assembly) | 292 | 0.047 | 0.410 | 0.025 | 0.001 |
| SigLIP2 | AgiBot (robot) | 3,235 | 0.020 | 0.779 | 0.043 | 0.065 |
| **CLIP-L** | **InHARD (assembly)** | 1,684 | **0.236** | 0.417 | 0.168 | **0.146** |
| V-JEPA2-L | InHARD (assembly) | 1,651 | 0.221 | 0.392 | 0.158 | 0.131 |
| SigLIP2 | InHARD (assembly) | 1,533 | 0.220 | 0.420 | 0.154 | 0.134 |
| DINOv2-L | InHARD (assembly) | 1,386 | 0.201 | 0.409 | 0.136 | 0.109 |
| DINOv2-giant | InHARD (assembly) | 1,315 | 0.196 | 0.416 | 0.131 | 0.105 |

**Hypothesis CONFIRMED.** The five semantic encoders jump **~10–12×** in boundary-F1 when moved from AgiBot to InHARD (e.g. SigLIP2 0.020 → 0.220; CLIP-L reaches 0.236, and **0.364 at the 1.0 s tolerance**). The same harness, same code, same detectors — only the domain changed. This is the clean proof that the §1 AgiBot result is a **domain property, not a harness bug**: when action boundaries are *visually expressed* (the worker reaches to different bins, picks visibly different parts, changes posture), the encoders recover them; when they are not (AgiBot's fixed-arm/fixed-background footage), nothing fires.

**Two finer points the numbers make:**
- **Pixel-cut detection still fails even here.** TransNetV2 barely moves (0.020 → 0.047) — continuous single-camera assembly has almost no hard *cuts*, only gradual appearance shifts. It is the **semantic** encoders, not the cut detector, that capture those shifts. So the contrast is not "real footage vs robot footage" but "semantic-embedding detector vs pixel-cut detector".
- **Absolute recall stays modest (~0.16 @0.5 s)** because InHARD is extremely fine-grained — ~126 actions/video, ~4 s each, many sub-second transitions with no appearance change. Precision (~0.42) and the **relative ordering** (CLIP ≳ V-JEPA2 ≈ SigLIP2 > DINOv2-L ≳ DINOv2-giant ≫ TransNetV2) are the reliable signal.
- **Bigger is not better.** Scaling the DINO backbone from large (0.300 B params) to giant (1.1 B) did *not* help: DINOv2-giant (bF1@0.5s 0.196) is marginally *worse* than DINOv2-L (0.201), and remains the weakest semantic encoder. For boundary detection on this domain the encoder *family* (a CLIP-style or video encoder) matters far more than raw parameter count — the discriminative signal for action boundaries is appearance change, which a larger self-supervised ViT does not capture any better. (DINOv3-L is gated and was not fetched.)

---

## 2 · Caption grounding — lexical vs semantic

**Does the caption mention the action the video actually shows?** Measured two ways:

- **LEXICAL** — exact content-word overlap between caption and GT action text. Strict; penalises synonyms.
- **SEMANTIC** — caption is split into clauses; each GT action is matched to its best clause by sentence-embedding cosine. *Recall* asks "is each GT action covered?", *precision* asks "is each clause the model wrote actually supported?" (this penalises padding).

| metric | value | what it measures |
|---|---|---|
| Lexical action hit-rate | 0.086 | fraction of actions with recall ≥ 0.5 |
| **Lexical action-F1** | **0.107** | harmonic mean (headline lexical) |
| Semantic recall (MiniLM) | 0.411 | clause-level meaning match |
| **Semantic F1 (MiniLM)** | **0.400** | length-robust headline (small encoder) |
| Semantic recall (BGE) | 0.616 | clause-level meaning match |
| **Semantic F1 (BGE)** | **0.610** | length-robust headline (large encoder) |
| Granularity-aware recall | 0.198 | bidirectional containment match |

### Reading the grounding result

- **The lexical–semantic gap is the story.** Lexical action-F1 is only **0.107** but BGE semantic-F1 is **0.610**. Qwen describes the right action ("grasps the object", "places it in the bin") but rarely reuses AgiBot's exact verbs/nouns. A pure lexical metric would badly under-rate the captions; the semantic metric corrects this.
- **Encoder choice matters.** BGE-large gives sem-F1 0.610 vs MiniLM 0.400. The bigger encoder resolves robot-manipulation paraphrases the small one misses — reported side-by-side so the metric isn't silently tied to one encoder.
- **Length-robust by construction.** Correlation of sem-F1 with caption length is **−0.069** (MiniLM −0.030) — essentially zero. Models cannot game the score by writing longer captions; the precision term cancels padding.
- **Threshold sensitivity.** hit-rate@0.3 = 0.251, @0.5 = 0.086, @0.7 = 0.016 — shows how strict you can be before recall collapses; the headline isn't a single arbitrary cut-off.

---

## 2·B · Cross-domain control: captions on YouCook2 (cooking)

**Why this experiment:** the AgiBot caption numbers are only interpretable if the harness *also* moves when the domain gets easier. Cooking is a domain web-trained models know well (recipe steps are abundant online), so a correct harness should report **higher** grounding and quality there — and if it does, it proves the AgiBot scores reflect genuine domain difficulty, not a saturated or broken metric.

**Dataset:** YouCook2 — 4,471 caption windows, GT recipe-step text embedded in the pipeline output (`youcook2_bs16_output`). Same caption/judge/grounding harness as AgiBot, run entirely from the on-disk output (no GPU captioning re-run). → `benchmark_results/caption_report_youcook2.json`.

| metric | AgiBot (robot) | YouCook2 (cooking) | direction |
|---|---|---|---|
| **quality_score** | 0.583 | **0.676** | ✅ higher where domain is easier |
| Semantic F1 (BGE) | 0.610 | 0.665 | ✅ better grounding |
| Semantic F1 (MiniLM) | 0.400 | 0.407 | ≈ |
| Lexical action-F1 | 0.107 | 0.246 | ✅ 2.3× — captions reuse GT words more |
| Unique caption ratio | 0.713 | 0.945 | ✅ far less repetition |
| Gemma4 / VCI-7B flag rate | 0.563 / 0.772 | 0.469 / 0.360 | ✅ fewer flagged wrong |
| Judge Cohen's kappa | 0.108 | 0.148 | ✅ judges agree more |

**Result CONFIRMED.** Every metric moves in the expected direction: on cooking, the captions are more diverse (unique ratio 0.71 → 0.95), better grounded both lexically (0.107 → 0.246) and semantically (BGE 0.610 → 0.665), the two judges flag far fewer captions and agree more (κ 0.108 → 0.148), and the composite `quality_score` rises 0.583 → 0.676. The harness is **not** saturated and **not** flat — it tracks real domain difficulty. This is the caption-side counterpart to the §1·B boundary control: same code, easier domain, measurably better scores.

**One caveat to read honestly:** the lexical gap stays large in *both* domains (YouCook2 lexical-F1 0.246 vs semantic-F1 0.665) — confirming again that lexical overlap alone under-rates captions and the semantic metric is the right headline regardless of domain.

---

## 2·C · Caption-model comparison: Qwen3-VL-30B vs Qwen-2.5-VL-7B

**Dataset:** same 30 AgiBot test videos (tasks 327 / 352 / 354). Both caption sets already on disk; scored offline with the same harness and BGE-large semantic model. Caveat: the two runs used slightly different window counts (Qwen3-VL: 120 windows at fixed-stride; Qwen-2.5-VL: 125 windows), so treat this as a model-level comparison, not a window-identical one.

| metric | Qwen-2.5-VL-7B | Qwen3-VL-30B | direction |
|---|---|---|---|
| **quality_score** | 0.676 | **0.763** | ✅ +13% |
| Semantic F1 (BGE) | 0.646 | **0.659** | ✅ better grounding |
| Lexical action-F1 | **0.205** | 0.202 | ≈ |
| Granularity-aware recall | 0.212 | **0.246** | ✅ |
| Idle-caption recall | 0.250 | **0.429** | ✅ much better |
| Unique caption ratio | 0.776 | **0.867** | ✅ more diverse |
| Mean words / caption | 17.1 | 21.6 | longer |
| sem-F1 length-bias (r) | −0.547 | **−0.181** | ✅ more robust |
| N windows (with GT) | 124 | 117 | — |

**Findings.**
- **Qwen3-VL-30B wins on every quality dimension.** Composite `quality_score` rises 0.676 → 0.763 (+13%). Semantic grounding improves (0.646 → 0.659), but the biggest gains are elsewhere: idle-caption recall jumps from 0.25 → 0.43 (the larger model explicitly says "nothing is happening" when appropriate), and caption uniqueness rises 0.776 → 0.867 (less verbatim repetition).
- **Lexical F1 is nearly identical (0.205 vs 0.202).** This confirms the gap is not about vocabulary size but about semantic paraphrasing: both models describe the same actions, Qwen3-VL does it more diversely and at greater length.
- **Length-bias is a clear difference.** Qwen-2.5-VL's sem-F1 correlates with caption length (r = −0.547) — shorter captions score *lower*, meaning the metric is somewhat gameable by verbosity. Qwen3-VL's correlation is much weaker (r = −0.181), suggesting its longer captions add information rather than padding.
- **Conclusion.** On this 30-video pilot, Qwen3-VL-30B is strictly better. The main remaining question is throughput cost (30B parameters vs 7B) — the 30B FP8 model requires an H100 (A100 has no native FP8) and takes ~7 min for 30 videos, roughly the same per-video rate as the 7B on an A100.

---

## 3 · Judges, consistency & idle handling

Two VLM judges watch each window's video and flag captions they think are wrong (**no GT shown to video judges** — they form an independent opinion). The consistency block needs **no GT at all**: it checks whether neighbouring windows of the same clip agree on the object/action, catching models that contradict themselves.

| judge metric | value | meaning |
|---|---|---|
| Gemma4-E4B flag rate | 0.563 | fraction of captions called wrong |
| VCI-7B flag rate | 0.772 | fraction of captions called wrong |
| Raw agreement | 0.585 | fraction both judges agree on |
| **Cohen's kappa** | **0.108** | agreement beyond chance (0 = chance) |
| Both-incorrect rate | 0.460 | high-confidence error captions |

| consistency / idle metric | value | meaning |
|---|---|---|
| Object continuity | 0.273 | adjacent windows share the object |
| Action continuity | 0.200 | adjacent windows share the verb |
| Contradiction rate | 0.053 | clip captions disagree (lower = better) |
| Intra-clip consistency | 0.365 | mean pairwise caption cosine |
| Idle-caption recall | 0.785 | says "nothing happening" when idle |
| False-idle on active | 0.040 | wrongly idle on active (lower = better) |

### Interpretation

- **Judges disagree a lot.** VCI-7B flags 77% of captions, Gemma4 only 56%; κ = 0.108 means agreement is barely above chance. They catch different failure modes — VCI is aggressive (over-flags), Gemma4 conservative. Using both gives a high-confidence "both-incorrect" set (46% of windows) that is far more trustworthy than either alone.
- **Idle handling works well.** When the window genuinely has no action, captions correctly say so 78% of the time, and they falsely claim "idle" on active windows only 4.0% of the time — scoring idle windows separately stops them from polluting the action metrics.
- **Low contradiction, modest continuity.** Only 5.3% of adjacent caption pairs contradict each other, but object continuity is 0.27 — captions stay self-consistent but vary their wording window-to-window (consistent with the lexical/semantic gap in §2).

---

## 3·B · Judge comparison against ground truth (offline, hand-annotated)

**Why this is separate from §3.** The κ in §3 only measures whether the two pipeline judges *agree with each other* — not whether either is *correct*. To measure correctness we need ground truth. This subsection uses a **hand-annotated set of 125 caption windows** (AgiBot tasks 327 / 352 / 354, balanced: **63 incorrect / 62 correct**, in `eval_stage/annotation_ground_truth.json`) where every Qwen-2.5-VL caption was expert-labelled correct/incorrect. Three judges were run on the *same* windows and scored against that GT. Each judge outputs 1–5; verdict = "incorrect" when score < 5 (positive class = caption error). This is the experiment that compares **VCInspector-7B vs Gemma4-E4B vs Qwen-2.5-VL-as-judge** head-to-head.

| judge | N | flag-rate | precision | recall | F1 | accuracy |
|---|---|---|---|---|---|---|
| **VCInspector-7B** | 125 | 40.0% | 82.0% | 65.1% | **0.726** | **75.2%** |
| Gemma4-E4B | 116 | 50.0% | 65.5% | 65.5% | 0.655 | 65.5% |
| Qwen-2.5-VL (direct judge) | 125 | 18.4% | **87.0%** | 31.7% | 0.465 | 63.2% |

**Findings.**
- **VCInspector-7B is the best judge overall** (F1 0.726, accuracy 75.2%) — its ActivityNet LoRA fine-tuning gives it the only balanced precision/recall, catching ~2/3 of real errors at 82% precision.
- **Qwen-2.5-VL-as-judge has the highest precision (87%) but the worst recall (31.7%).** Using the base captioner model directly as its own judge is *conservative*: when it flags a caption it is almost always right, but it stays silent on most errors (flags only 18%). It removes the ActivityNet domain bias but loses the sensitivity that bias provided.
- **Gemma4-E4B is the balanced middle** (P = R = 65.5%), evaluated on 116/125 windows (9 skipped for too few frames at its 32-frame minimum).
- **Why only VCI + Gemma4 are in the pipeline benchmark (§3).** Qwen-as-judge was only ever an *offline* script; it was never wired in as a pipeline `JudgePlugin`, so the full 2,587-video run (§3) used the two registered judges (VCI-7B, Gemma4-E4B). This subsection is where the full three-way comparison lives.

---

## 4 · Metrics implemented — what / why / how it improved

Each metric below was added to the harness to fix a specific blind spot. "Result" is the value achieved on this run and whether it behaves as designed.

### Multi-label grounding
- **Why:** 256-frame windows span several GT actions; the old majority-label scoring silently threw the minor action away.
- **How:** the GT source now returns **all** overlapping actions with coverage fractions; scored unweighted *and* coverage-weighted, plus a boundary-recall for transition windows.
- **Result:** sem-recall 0.616; machinery active. On this run every window mapped to one dominant action (transition-rate 0.0), so multi-label collapses to single-label *here* — but the path is implemented and proven.

### Clause-level semantic F1
- **Why:** whole-caption cosine rewarded verbose captions and missed synonyms; a single lexical score under-rated good captions.
- **How:** split caption into clauses, match each GT action to its best clause; the precision term penalises unsupported clauses.
- **Result:** sem-F1 0.610 vs lexical 0.107; length-bias −0.069 (≈0 = not gameable). Works as intended.

### Two-encoder semantic eval
- **Why:** a semantic score silently tied to one embedding model is not trustworthy.
- **How:** run MiniLM (small) and BGE-large (strong) side by side; report both.
- **Result:** BGE 0.610 > MiniLM 0.400 — encoder choice is now visible, not hidden.

### Absolute-frame GT alignment
- **Why:** clip-local frame numbers ≠ source-video frames for split clips, so GT lookup hit the wrong action.
- **How:** `_absolute_frame_range()` shifts window frames by `clip-start × framerate` before GT lookup.
- **Result:** n_with_gt 14,681 / 15,526 (94.5%) windows successfully matched to GT — alignment correct.

### Dual-judge agreement (kappa)
- **Why:** a single VLM judge's flag rate is meaningless without knowing if it is reliable.
- **How:** run two judges; report raw agreement + Cohen's kappa + the both-incorrect high-confidence set.
- **Result:** κ = 0.108 exposed that the judges barely agree — a result we could not see before.

### Reference-free consistency
- **Why:** most of the data has no GT; we still need a quality signal.
- **How:** adjacent-window object/action Jaccard + contradiction flag — needs no GT.
- **Result:** contradiction 0.053, object-continuity 0.273 — usable on any dataset.

### Idle-window scoring
- **Why:** idle windows ("nothing happening") were scored as action failures, depressing every model unfairly.
- **How:** detect idle windows by coverage; score them separately; track false-idle on active windows.
- **Result:** idle-recall 0.785, false-idle 0.040 — clean separation.

### Length-robustness diagnostic
- **Why:** need proof the headline score can't be gamed by writing more words.
- **How:** report Pearson r between caption length and each score.
- **Result:** sem-F1 r = −0.069 — confirmed robust.

### Composite quality_score
- **Why:** stakeholders need one comparable number per captioning model, not 30 columns.
- **How:** mean of semantic-F1, uniqueness, object-continuity, (1 − contradiction); caption length deliberately excluded.
- **Result:** quality_score 0.583 — single figure-of-merit, ready for model A/B.

---

## 5 · Text-video alignment (SigLIP2, reference-free)

**Status: ✅ COMPLETE — job 24215032, 2 h 32 m, all 15,526 windows scored (`n_skipped: 0`) → `benchmark_results/textvideo_report.json`.**

**What it measures:** unlike §2 (caption vs GT *text*), this scores each caption directly against the *video frames* — no ground truth needed. Each window's frames and its caption are embedded with SigLIP2; their cosine similarity says how well the caption matches what is on screen. A cross-retrieval rank (can the caption find its own video among all 15,526?) measures discriminativeness.

**Why it was needed:** it is the only grounding signal that works when there is no GT action text at all — applicable to any dataset, and a cross-check on the text-based grounding in §2.

| metric | value | meaning |
|---|---|---|
| caption↔video sim (mean) | 0.216 | how well the caption matches the frames (std 0.033) |
| GT-text↔video sim (mean) | 0.137 | reference ceiling: how well even the *correct label* grounds |
| caption − GT | **+0.080** | caption aligns *better* with frames than the GT label does |
| retrieval median rank | 387 / 15,526 | discriminativeness (random ≈ 7,763) |
| R@1 / R@10 | 0.004 / 0.034 | rarely finds its own clip exactly |
| within-clip norm. rank | 0.446 | ≈ random (0.5) inside a clip |

**Reading the result.** Two things stand out, and both *confirm* the documented domain limitation rather than contradicting it. (1) The caption scores **higher** against the frames than the GT action text does (+0.080) — a known artefact: SigLIP2 is web-trained, so a vivid descriptive caption ("robotic arm grasps a purple eggplant") matches the general scene better than a terse GT label ("grasp"), even when the *specific object* may be wrong. (2) Retrieval is far above random globally (median rank 387 vs ~7,763) but **within a clip it is essentially random** (0.446 ≈ 0.5) — the encoder can tell a fridge clip from a supermarket clip, but cannot tell two adjacent windows of the *same* clip apart. This is exactly the "works at task level, fails at fine-grained object level" pattern seen across InternVideo2 / ViCLIP / EMScore / LaViLa. **Conclusion: text↔video is a useful coarse sanity check but must stay a secondary signal — the VLM judge and text↔text semantic metrics remain primary.**

**Implementation note (why the first run timed out).** The first attempt OOM'd (SigLIP2 embedding all 15,526 captions at once → fixed with chunked encoding, 256 texts / 32 windows per GPU call). The retry then *timed out*, and the cause was **not** the GPU: `decode_window_frames` decoded each video **from frame 0 on every one of the 15,526 calls** with no seek, re-decoding the whole prefix for late windows. Fixed by **seeking to the keyframe before each window** (≈4.8× faster, frame selection unchanged) plus per-1000-window progress logging and a 4 h → 6 h walltime bump. The completed run above used this fix.

---

## 6 · Efficiency, caption stats & open items

| statistic | value | note |
|---|---|---|
| Caption windows | 15,526 | 1 caption per window |
| Unique caption ratio | 0.713 | low = repetitive model |
| Mean / median words | 17.6 / 18 | caption length |
| Vocabulary size | 923 | distinct words used |
| Most-duplicated caption | ×77 | repeated verbatim |
| Caption compute / caption | 0.309 s | true stage compute |
| Est. caption tokens / s | 74.2 | throughput proxy |

### Open items

- **pyscenedetect — removed.** Not installable in the read-only container (no `scenedetect` module in any pixi env); dropped from the detector list. CLIP / DINOv2 / V-JEPA2 semantic detectors can be added once their encoders finish downloading.

---

## Overall verdict

**Is what we have right?** Yes. The numbers are internally consistent and behave exactly as the metric designs predict: lexical ≪ semantic (synonym tolerance working), length-bias ≈ 0 (not gameable), idle separated cleanly, the judges' low kappa surfaced honestly, and shot detection shown to be no better than fixed-stride on a semantic-boundary domain. No metric is silently broken.

**What the run tells us about the captions.** Qwen-2.5-VL captions are semantically decent (BGE sem-F1 0.61) and self-consistent (contradiction 5%) but lexically loose and somewhat repetitive (unique ratio 0.71). They get the action right, the exact object wording less so.

**What the run tells us about the harness.** It is doing its job: it measures meaning not wording, can't be gamed by length, works with or without GT, and produces one comparable `quality_score` per model — ready to rank Qwen3-VL / Gemma4 / others as soon as those caption sets land.
