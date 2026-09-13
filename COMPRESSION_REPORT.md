# Compression report: extended record → concise submission

Date: 13 September 2026. Branch `claude/thesis-compression-concise-76c12d`.
The extended thesis is preserved unchanged at the git tag and branch
`extended-thesis-pre-compression` (commit `9ab1b87`) and as
`archive/extended-thesis-230pp.pdf`. Recover its sources with
`git checkout extended-thesis-pre-compression`.

## Before

| Part | Physical pages | Count |
| --- | --- | --- |
| Mandatory front pages | 1–4 | 4 |
| Table of contents | 5–10 | 6 |
| AI disclosure | 11 | 1 |
| 1 Introduction | 12–19 | 8 |
| 2 Statistical mechanics, EBMs, diffusion | 20–44 | 25 |
| 3 The model, and the object of study | 45–58 | 14 |
| 4 The rotating ring | 59–72 | 14 |
| 5 Gaussian chain by linear algebra | 73–102 | 30 |
| 6 Gaussian chain by belief propagation | 103–116 | 14 |
| 7 Beyond Gaussianity | 117–133 | 17 |
| 8 EM on the chain | 134–155 | 22 |
| 9 Learning the kernel / convergence | 156–165 | 10 |
| 10 What structure buys | 166–200 | 35 |
| 11 Discussion and conclusions | 201–206 | 6 |
| Bibliography | 207–215 | 9 |
| Acknowledgements | 216 | 1 |
| Appendix A Reproducibility | 217–225 | 9 |
| Appendix B Aggregation | 226–230 | 5 |
| **Total** | | **230** (3.00 MB) |

## After

| Part | Physical pages | Count | Budget in the brief |
| --- | --- | --- | --- |
| Mandatory front pages | 1–4 | 4 | 4 |
| Table of contents | 5–6 | 2 | 2–3 (with disclosure) |
| AI disclosure | 7 | 1 (half a page of text) | ≤ ½ page |
| 1 Introduction | 8–9 | 2 | ≤ 2 |
| 2 Statistical mechanics, EBMs, diffusion, related work | 10–17 | 8 | 5–10 |
| 3 The Gaussian chain by linear algebra | 18–27 | 10 | 15–18 for Ch. 3+4 |
| 4 The Gaussian chain by belief propagation | 28–37 | 10 | (20 achieved) |
| 5 Beyond Gaussianity | 38–47 | 10 | 8–10 |
| 6 Learning the parameters: EM | 48–56 | 9 | 7–8 |
| 7 Learning the kernel, and what convergence costs | 57–63 | 7 | 5–6 |
| 8 What structure buys | 64–83 | 20 | 15–18 |
| 9 Discussion and conclusions | 84–85 | 2 | ≤ 2 |
| Bibliography | 86–92 | 7 | natural |
| Acknowledgements | 93 | 1 | – |
| Appendix A Reproducibility (incl. aggregation) | 94–97 | 4 | ≤ 2 |
| **Total** | | **97** (1.03 MB) | 90–100 |

Chapters 3+4, Chapter 8 and the appendix each run about two pages over the
brief's individual targets; the total is inside the 90–100 window and under
100. The overrun in Chapters 3, 4 and 8 is float area, not prose: every
project-generated figure and table was kept (20 figures, 19 tables), and
the appendix carries the figure-to-code map and the generated aggregation
table. The bibliography was not touched for length (69 printed entries, all
cited; 1.5 line spacing kept for consistency with the body).

## Major structural changes

- **Old Chapter 3 deleted as a standalone chapter.** Its essential content was
  redistributed: the AR(1)/OU model definition (one third of a page) and the
  `d = 2` joint-versus-marginal example (half a page, the boxed two-frame
  score) open new Chapter 3; the factor-graph definition, the sum–product
  rules, the tree-exactness theorem (Mézard–Montanari Thm 14.1, with a
  three-line proof sketch), the reproduced BP-rules figure and the TikZ
  posterior factor graph open new Chapter 4. Dropped: stationarity tutorial,
  Markov-chain definitions, the OU integrating-factor toolbox, the
  research-history and propagator-false-start sections, the MRF/RBM material.
- **Old Chapter 4 (rotating ring) deleted entirely**, with its three
  figures. Every forward and backward reference was removed from the
  introduction, roadmap, contributions, conclusions, limitations, captions
  and the appendix figure map; the ring is not presented as a precursor.
  A global search for ring / rotation / gauge / radial / circle / planar in
  the surviving sources finds only the eigenbasis "rotation" of the spectral
  theorem, which is unrelated.
- **Renumbering:** 11 chapters → 9; two appendices → one. Every internal
  reference is a `\ref`/`\eqref`; no chapter or theorem number is typed
  literally. Old chapter labels that other chapters used (`ch:model`,
  `ch:graphs`, `sec:factor-graphs`, `sec:sum-product`, `sec:ou`, …) are
  kept as aliases on the sections that now carry the material, so nothing
  dangles.
- **Toolbox derivations removed** (13 boxes: max-entropy, EBM gradient,
  Fokker–Planck, Tweedie step-by-step, OU integrating factor, reading the
  precision off the log-density, completing the square, the two-sweep
  proof, Gaussian closure, band fill, transfer-matrix inversion, …). Each is
  now a stated result with a 3–10-line proof or a citation. The `tcolorbox`
  and `listings` packages are no longer loaded.
- **Per-chapter compression:** introduction 8 → 2 pp (motivation and gap,
  RQ1–RQ6 unchanged in content, three contributions, one-sentence roadmap;
  overview figure moved to the opening of Chapter 3); background 25 → 8 pp
  (path-measure/ELBO/Jarzynski section and six reproduced literature
  figures removed, Tweedie derivation collapsed to one sentence, three
  scores kept, transitions and locality condensed, taxonomy reduced to one
  paragraph, gap statement kept); Gaussian chapters 44 → 20 pp (no
  unrolled recursions, no entrywise covariance algebra, spectral section
  reduced to one theorem, sub-subsections merged); Laplace 17 → 10 pp
  (Theorem on LMMSE with a short induction proof, grid solver choices as a
  table); EM 22 → 9 pp (model, decomposition, one boxed algorithm,
  Fisher's identity in one display, boxed updates); convergence 10 → 7 pp;
  structure comparison 35 → 20 pp (bulk posterior as one proposition with
  a five-line proof, cumulant argument as result–prediction–measurement,
  summary table folded into a paragraph, δ table folded into text);
  conclusions 6 → 2 pp.
- **Appendices merged** into one reproducibility appendix: figure/table-to-
  code map, provenance routes, protocol facts (all numbers through the
  generated macros), environment, and the aggregation robustness statement
  with the generated structured-aggregation table.
- **Typography:** body font switched to Arial (fontspec under tectonic's
  XeTeX; falls back to TeX Gyre Heros if Arial is absent), as the Bocconi
  guide recommends. Chapter-head white space reduced (50/40 pt → 20/30 pt)
  and section-head skips reduced by about a third; contents list set at
  single spacing. Font size, margins and body line spacing are unchanged.

## Scientific preservation

- **Figures:** all 20 project-generated figures that survive the chapter
  deletions are present (eigen-relax, sparsity lifecycle, Markov survival,
  truncation, Laplace nonlinearity/metrics/closure, EM grid/diagnostics/
  innovation, two rates, value of structure, screening, capacity,
  non-Markov, information, forward corruption, pipeline and posterior factor
  graph schematics) plus the one reproduced BP-rules figure. Removed:
  three rotating-ring figures (deleted chapter), six literature
  reproductions (background budget), and the marginal-blindness
  illustration (its content is the boxed two-frame score).
- **Tables:** all generated tables are input unchanged from `sections/`
  (efficiency, structured, screening, capacity, non-Markov, convergence,
  structured aggregation) and all hand-typed result tables are kept
  (truncation, closure sweep, grid budgets, EM checks, recovery, rates,
  channel cost, protocol, architecture ladder, oracles, correlation length,
  information loss). One summary table (chapter-8 verdicts) and one
  duplicate-column table (δ values, now in the text) were removed.
- **Numbers:** no reported value, sample size, error bar or setting was
  edited; every experiment number is still a generated macro, and
  `check.sh` verifies the abstract against those macros. No experiment was
  rerun.
- **Caveats retained:** scope conditions on RQ6 (asymmetric information
  in both directions, no causal decomposition, ratios at fixed data), the
  aggregation dependence of the headline, the iteration-cap and
  resolution-floor caveats on capacity, the exact-KL-only-for-Gaussian
  caveat on δ, the ECM-not-exact-EM caveat on the missing-information
  identity, the coordinate-slope-not-eigenvalue caveat, the delta-method
  status of the information prediction, the "measured regularity, not a
  law" reading of the δ collapse, and the withdrawn generative comparison.
- **Theorems:** statements of Theorem 5.2 (LMMSE), Proposition 8.2 (bulk
  covariance), Theorem 3.8 (band fill), Theorem 4.3 (Gaussian closure) and
  Proposition 8.3 (rank-one perturbation) are unchanged; only their proofs
  were shortened. No Lean or other formal artefacts live in this
  repository, so none needed rerunning.

## Bocconi compliance

| Requirement | Status |
| --- | --- |
| Pages 1, 2, 4 blank | yes (`pdftotext` returns nothing; no header/footer) |
| Page 3 dedication only | yes: "To my family." only, `empty` page style |
| Text and numbering begin after page 4 | yes; page 5 is the Contents, numbered 5 |
| No title page, abstract, name, ID, seal | yes (`abstract.tex` is not input; no personal identifier in the PDF text) |
| Right-hand start of the body | yes: Contents on p. 5, disclosure on p. 7, Chapter 1 on p. 8 after an odd-page check |
| A4 | 595.28 × 841.89 pt |
| Left/right margins 2.5 cm | `hmargin=2.5cm` in `preamble.tex` |
| 12 pt body | `\documentclass[12pt]`, Arial (recommended family) |
| 26–30 lines per prose page | `\setstretch{1.5}`, text height 630 pt ≈ 29 lines; measured 27–31 text lines on prose-heavy pages (9, 12, 16, 20, 33, 68, 85) |
| Page numbers | running head (right) on text pages, footer on chapter openings |
| Single PDF < 10 MB | 1.03 MB |
| AI use disclosed and referenced | half-page disclosure; Claude, ChatGPT, Gemini cited in the bibliography |

## Build verification

- Command: `./check.sh` (runs `tectonic -X compile main.tex --keep-intermediates --keep-logs --reruns 4`, then inspects log and PDF).
- Result: `clean: 97 pp, 20 figures, 19 tables, 69 citations`.
- Undefined references: 0. Undefined citations: 0. Multiply-defined labels: 0. `??` in the PDF: 0. Missing glyphs: 0. Stray source tokens: 0. Overfull boxes over 10 pt: 0 (six under 10 pt, worst 9.8 pt). Abstract-vs-macro check: 0 mismatches. PDF title matches `title.tex`.
- Build manifest regenerated (`tools/make_build_manifest.py`): 19 generated inputs, 18 figure files.
- Research tests relevant to the manuscript: `python3.12 -m pytest research/tests/{test_gaussian_bp_equivalence,test_gaussian_precision_score,test_recursion_agreement,test_message_normalization,test_em_bp,test_fisher_bruteforce,test_grid_convergence,test_nonmarkov,test_depth_law,test_bp_mixture}.py -m "not slow"` → 100 passed in 89 s.
- Visual inspection: pages 1–5, 8 (introduction), 24 (spectral form), 32 (Gaussian closure), 70 (structure comparison), 85 (conclusions) and 94 (appendix) rendered and checked.

## Git summary

- Preserved: tag + branch `extended-thesis-pre-compression` at `9ab1b87`; `archive/extended-thesis-230pp.pdf`.
- Deleted: `chapters/ch02-statmech-diffusion.tex`, `ch03-model.tex`, `ch04-ring.tex`, `ch05-gaussian-matrix.tex`, `ch06-gaussian-bp.tex`, `ch07-laplace.tex`, `ch08-em-parameters.tex`, `ch09-em-kernel.tex`, `ch10-comparison.tex`, `ch11-conclusions.tex`, `appB-aggregation.tex`; figures `fig_ring_*.pdf` (3), `fig_song_dog.png`, `fig_song_sde.png`, `fig_bachtis_cascade.png`, `fig_bonnaire_memorize.png`, `fig_biroli_regimes.png`, `fig_unet_architecture.png`, `fig_marginal_blindness.pdf`.
- Added: `chapters/ch02-background.tex`, `ch03-gaussian-matrix.tex`, `ch04-gaussian-bp.tex`, `ch05-laplace.tex`, `ch06-em-parameters.tex`, `ch07-em-kernel.tex`, `ch08-comparison.tex`, `ch09-conclusions.tex`; `archive/`; this report.
- Rewritten: `main.tex`, `preamble.tex`, `chapters/disclosure.tex`, `chapters/ch01-introduction.tex`, `chapters/appA-reproducibility.tex`, `README.md`; `references.bib` gained three AI-system entries; `sections/build-manifest.tex` regenerated; `main.pdf` rebuilt.
- Untouched: `sections/*-numbers.tex`, `sections/tab-*.tex`, `figures/*` that remain, `tools/`, `research/`, `abstract.tex`, `title.tex`, `check.sh`, `notation.tex`, `chapters/acknowledgements.tex`.
