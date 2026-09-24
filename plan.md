# Research Plan: Language Drift in Depth-Pruned Multilingual LLMs

*Working title options:*
- "Pruned in English, Broken in Hindi: Language Drift as a Hidden Cost of Depth Pruning"
- "Does Block Influence Speak Your Language? Calibration-Language Effects in LLM Depth Pruning"
- "Lost in Layers: Depth Pruning Silently Destroys Multilingual Generation"

---

## 1. The core claim

Depth pruning methods (ShortGPT, LaCo, Gromov et al.) select layers using an
importance metric computed on **English** calibration data, and validate on
**English** benchmarks — predominantly multiple-choice. We show:

1. The layer ranking produced by Block Influence (BI) is not language-invariant,
   contradicting ShortGPT's robustness claim (Appendix A, Table 10), which tested
   only PG19 vs MMLU — both English.
2. English-calibrated pruning damages Indic-language performance substantially
   more than English performance at identical pruning ratios.
3. A specific, previously unmeasured failure mode: **language drift**. Pruned
   models progressively stop responding in the input language, reverting to
   English or emitting mixed script. Single-letter MCQ evaluation cannot detect
   this by construction.
4. A cheap mitigation: protecting the final *k* layers (the "language control"
   band identified by LinguaMap) recovers language fidelity at the same
   compression ratio.

**Why claim 3 is the paper.** Claims 1–2 are a benchmark extension. Claim 3 is a
new failure mode with a new metric, and it explains why English-centric pruning
evaluation has missed this: MMLU reads four scalars at one position, so a model
that has lost the ability to *produce* Hindi still scores fine on Hindi MMLU.

---

## 2. Positioning against prior work

| Prior work | What it did | Why it isn't this paper |
|---|---|---|
| ShortGPT (Men et al., ACL Findings 2025) | BI metric, depth pruning; ShortGPT-gen for generative tasks | Zero multilingual evaluation. Calibration robustness tested on PG19 vs MMLU only |
| Kurz et al. (TACL, arXiv 2408.14398) | First study of calibration language for multilingual pruning | Wanda/SparseGPT (weight sparsity), not depth pruning. AR/DE/EN/ES/RU/SW/ZH — no Indic |
| L3Cube (arXiv 2501.00733, 2409.14168) | Layer pruning for low-resource languages incl. Indic | Encoder models (BERT/sentence-BERT), not decoder LLMs; no BI |
| LinguaMap (ICLR 2026) | Localizes language control to final few layers; tuning 3–5% of params lifts language consistency from <20% to >98% | Fine-tuning, not pruning. Supplies our mechanism and our mitigation |
| Wendler et al. (ACL 2024), Tang/Wang/Zhao 2024 | Input/output layers language-specific, middle layers language-agnostic | Interpretability only; no compression |
| Cross-Layer Transcoders (arXiv 2511.10840) | Tokenizer bias forces early layers into word reassembly for non-Latin scripts | Explains our fertility hypothesis; no pruning |
| Rethinking Layer Redundancy (arXiv 2604.24938) | Redundancy depends on calibration configuration, not intrinsic structure | Supports our premise; English-only |
| Gromov et al. 2024 | Angular distance over layer blocks | English-only |
| LoRP (arXiv 2605.27786) | Representation Locality Score, cluster-aware budget allocation | English-only; possible additional baseline |

**Novelty statement for the paper:** to our knowledge, no prior work evaluates
depth pruning of decoder-only LLMs on Indic languages, tests whether BI's layer
ranking transfers across languages, or measures output-language fidelity as a
pruning metric.

**Caveat:** this field moves fast and my search was not exhaustive. Before
writing, re-search: `depth pruning multilingual`, `layer pruning Indic LLM`,
`language drift pruning`, `block influence multilingual`. Check the
Awesome-LLMs-Pruning repo (github.com/liyunqianggyn/Awesome-LLMs-Pruning) and
recent MRL/LoResLM workshop proceedings.

---

## 3. Experimental design

### 3.1 Language selection (3 languages, controlled)

| Language | Script | Family | Rationale |
|---|---|---|---|
| **Hindi** | Devanagari | Indo-Aryan | Highest-resource Indic; best chance of non-trivial dense baseline |
| **Marathi** | Devanagari | Indo-Aryan | *Same script as Hindi, lower resource* — isolates resource level from script |
| **Tamil** | Tamil | Dravidian | Different script and family; agglutinative; high tokenizer fertility |

This gives a near-controlled design: Hindi↔Marathi varies resource level holding
script constant; Hindi↔Tamil varies script and typology. If you must swap one,
Telugu (Dravidian, own script) or Bengali (Indo-Aryan, own script) work; avoid
picking three languages that differ on everything at once, since you then can't
attribute anything.

### 3.2 Models

Pick 3, spanning English-centric to Indic-native. The contrast is itself a finding.

| Model | Size | Layers | Why |
|---|---|---|---|
| **Sarvam-1** | 2B | — | Indic-native pretraining, 10 Indic languages, low tokenizer fertility for Indic |
| **Gemma-2-2B** or **Gemma-3-4B** | 2–4B | 26 / 34 | Genuinely multilingual, widely used, fits on free-tier GPUs |
| **Llama-3.1-8B** | 8B | 32 | English-centric baseline; standard in the pruning literature |

Optionally add **Qwen2.5-7B** (different tokenizer, different multilingual mix).
Avoid Llama-2 — its Indic ability is too weak, and floor effects will destroy
your retention ratios.

**Sanity gate before anything else:** verify each dense model scores
meaningfully above chance/floor on each language and task. If dense Tamil XL-Sum
ROUGE is ~2, you cannot measure degradation. Report this table; it justifies your
model selection and pre-empts the obvious reviewer objection.

### 3.3 Calibration data — use parallel corpora

This is a key design choice. **FLORES-200 dev** is parallel across all your
languages: identical semantic content, different language. That isolates the
language effect from content effect, which no prior calibration study has done
cleanly.

| Set | Source | Use |
|---|---|---|
| English | FLORES-200 dev (eng_Latn) | Baseline calibration |
| Hindi / Marathi / Tamil | FLORES-200 dev (hin_Deva, mar_Deva, tam_Taml) | Target-language calibration |
| Mixed | Equal sample across all four | Multilingual calibration condition |
| PG19 | (ShortGPT's original) | Reproduction check only |

Supplement with **Sangraha** (AI4Bharat) or **IndicCorp v2** if you need longer
sequences — FLORES sentences are short, and BI is averaged over token positions.
Use ~256–512 sequences of ~1k tokens per condition; match token counts across
languages, not sentence counts (fertility differs).

### 3.4 Evaluation suite

Split deliberately by readout protocol, because that's central to the argument.

**Multiple-choice (single forward pass):**
- MILU or IndicMMLU-Pro — Indic MMLU analogue
- Belebele — reading comprehension, 122 languages, all three of yours
- IndicXNLI — NLI, 11 Indic languages
- IndicCOPA (from IndicXTREME)

**Generative (autoregressive):**
- XL-Sum (hi, mr, ta subsets) — summarization, ROUGE. Direct analogue of
  ShortGPT's XSum, so results are comparable to their Table 3
- FLORES-200 translation (en→xx and xx→en) — chrF++ and spBLEU
- IndicQA or XQuAD-IN (from IndicGenBench) — extractive/short-form QA, F1
- IndicGenBench CrossSum-IN — cross-lingual summarization

Keep English versions of each as the control.

---

## 4. Metrics

### 4.1 Standard
- Perplexity per language (teacher-forced)
- Task metric per benchmark
- **Retention ratio** per language = pruned / dense, following ShortGPT's `Per.` column

### 4.2 Novel — the contribution

**(a) Language Identification Accuracy (LID).** Run **GlotLID** (preferred over
fastText lid.176 for Indic coverage) on every generated output. Report the
fraction generated in the target language.
- Filter generations under ~20 characters; LID is unreliable on short strings
- Report LID confidence distribution, not just argmax
- Also report the *destination* language when drift occurs — is it English, or
  the wrong Indic language? Hindi↔Marathi confusion would be its own finding

**(b) Script fidelity.** Fraction of characters in the expected Unicode block
(Devanagari U+0900–U+097F, Tamil U+0B80–U+0BFF). Catches mixed-script output
that LID may score as target-language.

**(c) BI rank agreement across calibration languages.** For each model, compute
BI under each calibration language, then report:
- Spearman ρ and Kendall τ between the full layer rankings
- **Jaccard overlap of the top-*k* removal sets** at k = 12.5%, 25%, 37.5%
- Per-layer BI variance across languages, plotted against layer index

This is the direct test of ShortGPT's robustness claim. Jaccard is the metric
that matters practically — ρ can be 0.95 while the actual removal set differs.

**(d) MCQ decision margin.** m = score(correct) − max score(wrong). Accuracy is
a step function that hides pre-flip degradation; margin is continuous. Predict
the margin shrinks substantially in Indic before accuracy moves — which is
precisely the signal that connects MCQ resilience to generative collapse.

**(e) Tokenizer fertility as covariate.** Tokens-per-word per language per model.
Test whether fertility predicts pruning damage, and whether it predicts a shift
in the *shallow* boundary of the safe removal band.

**(f) Per-step KL divergence.** KL(dense ‖ pruned) at generation step *t* under
the pruned model's own rollout, per language. Tests whether error accumulation is
faster in Indic. Use this instead of ROUGE for the compounding analysis — ROUGE
floors at ~0 and carries no information about relative severity below that point.

---

## 5. Phased plan

### Phase 0 — Infrastructure and reproduction (1–2 weeks)
- Clone `github.com/icip-cas/ShortGPT` (official) and/or `short-transformers`
  (melisa-writer), which already implements BI, angular distance, and linear
  approximation. Do not reimplement.
- Reproduce ShortGPT's Llama2-7B removal set {21…29} on PG19. If you don't
  recover it, your BI implementation is wrong — stop and fix before proceeding.
- Wire up lm-eval-harness for MCQ; write the generation harness yourself with
  the alignment care discussed separately (position `j−1` predicts token `j`).
- **Gate:** dense baselines on all 3 models × 4 languages × all tasks, confirming
  non-trivial scores.

### Phase 1 — Is BI language-invariant? (2 weeks)
- Compute BI for 3 models × 5 calibration conditions (en, hi, mr, ta, mixed)
  using FLORES-200 dev, matched token counts.
- Deliver: overlaid BI curves per model; ρ/τ/Jaccard tables; per-layer variance plot.
- **Two outcomes, both publishable.** If rankings diverge → ShortGPT's robustness
  claim fails for non-English, and target-language calibration matters. If
  rankings agree → the claim holds, which makes claim 3 *more* interesting,
  because the damage then isn't attributable to picking the wrong layers.

### Phase 2 — Cross-lingual damage under English calibration (2 weeks)
- Prune each model at 12.5% / 25% / 37.5% using **English** calibration (the
  field default). Layer deletion is free — no training, just slice the ModuleList.
- Evaluate all tasks × all languages.
- Deliver: retention-ratio table, English vs each Indic language, per ratio.
  Plot retention vs ratio, one line per language.
- **Hypothesis:** Indic retention curves fall off earlier and steeper, and the
  MCQ/generative gap is wider in Indic than in English.

### Phase 3 — Language drift (1–2 weeks) — **the headline**
- Run GlotLID + script fidelity on every generation from Phase 2.
- Deliver: LID accuracy vs pruning ratio, per language per model. Drift
  destination breakdown. Qualitative examples table (Hindi prompt → English
  answer) — reviewers respond to these.
- Cross-tabulate: at the ratio where LID collapses, what is MMLU/Belebele
  accuracy? If accuracy is still high while LID has collapsed, you have
  demonstrated that MCQ evaluation is blind to this failure. **That is the paper's
  money figure.**

### Phase 4 — Does target-language calibration help? (1 week)
- Prune with Hindi calibration → evaluate Hindi. Same for Marathi, Tamil. Plus
  the mixed condition.
- Compare against English-calibrated at matched ratio.
- **Prior expectation:** Kurz et al. found target-language calibration lowers
  perplexity but does not reliably help downstream tasks. Test whether that
  transfers to *depth* pruning. A different answer for depth vs weight pruning is
  itself a contribution.

### Phase 5 — Mitigation (2 weeks)
Three options, in increasing cost:
1. **Protect the tail.** Exclude the final *k* layers (k = 2, 3, 4) from the
   removal pool, motivated by LinguaMap's localization of language control.
   Measure LID recovery at matched compression. Cheapest possible fix; if it
   works, it's a clean practical recommendation.
2. **ShortGPT-gen comparison.** It routes generated tokens through all layers, so
   it should fix drift by construction — verify, and note it doesn't reduce model
   size, which is the trade-off your Option 1 avoids.
3. **LoRA recovery** on a small Indic instruction set (IndicAlign, Sarvam
   instruction data). Only if time permits; adds a training dependency.

### Phase 6 — Write-up (2 weeks)

**Total: ~10–12 weeks** at part-time intensity.

---

## 6. Compute budget

- BI computation: minutes per condition. 3 × 5 = 15 runs. Negligible.
- Pruned checkpoints: 3 models × 3 ratios × 5 calibrations = 45 configs, but
  layer deletion is free and requires no storage — generate on the fly.
- **Evaluation is the entire cost.** Generative eval dominates. Rough estimate:
  a few hundred GPU-hours on A100/L4 for the full grid.
- 2–4B models fit comfortably on Colab Pro+, Kaggle T4×2, or a single L4.
  Llama-3.1-8B needs A100 40GB or 4-bit quantization (note: ShortGPT is
  orthogonal to quantization per their Table 7, so 4-bit eval is defensible —
  but report it as a separate condition, not silently).
- **Cut the grid if needed:** drop to 2 models, or evaluate the full grid on MCQ
  but only 25% ratio on generative. Do not drop the LID measurement — it's the paper.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Floor effects** — dense Indic performance too weak to measure degradation | Phase 0 gate. Prefer Indic-native models. Report dense baselines prominently |
| **LID unreliable on short/degenerate output** | GlotLID over fastText; length filter; report confidence; manually verify a 100-sample subset and report agreement |
| **Scooped** — fast-moving field | Move fast; frame around the drift metric rather than "we evaluated on Indic," which is much easier to scoop |
| **Null result on Phase 1** (BI is language-invariant) | This strengthens Phase 3 rather than killing the paper. Frame Phase 1 as a validation, not the contribution |
| **ShortGPT-gen already fixes it** | Likely true for drift, but it doesn't shrink the model. Your Option-1 mitigation preserves the size reduction, which is the whole point of pruning |
| **Reviewer: "why Indic specifically?"** | Answer with the mechanism, not with representation: high tokenizer fertility + non-Latin script + language-specific tail layers. Indic is a *stress test* for the mechanism, not a diversity checkbox |
| **Hindi↔Marathi confounded by shared script** | That's a feature — it's the controlled comparison. State it as such |

---

## 8. Venue strategy

**Workshop first** (faster feedback, lower risk of scoop while you iterate):
- **MRL** (Multilingual Representation Learning) @ EMNLP — best fit
- **LoResLM** — low-resource language models
- **WiNLP**
- **ICON** — Indian NLP conference; strong regional visibility

**Then a short paper:** ACL / EMNLP short, or COLING. The drift metric plus the
mitigation is enough for 4 pages. Don't try for a long paper on this scope.

**Also:** post the arXiv preprint early once Phase 3 is solid. L3Cube's Indic
pruning papers follow exactly this route.

---

## 9. What to nail down in week 1

1. Re-run the novelty search. Confirm nobody has published this in the last
   few months.
2. Verify dense baselines are non-trivial for your 3 models × 3 languages.
   **This decides whether the paper is possible at all.**
3. Reproduce {21…29} on Llama2-7B to validate your BI pipeline.
4. Pick GlotLID vs alternatives and validate it on FLORES references (known
   ground truth) — establish your LID measurement's own accuracy ceiling before
   you use it to measure anything else.
