# BP-informed diffusion scores for (approximately) Markovian sequences

Continuation of the `bp_markov_diffusion_gaussian_approx` project. Research question:

> How can (approximate) Markovianity of sequence data be exploited to compute the
> diffusion score in a more informed way than a black-box network, and what does
> this reveal about reverse-diffusion dynamics?

## Headline results of this iteration

1. **Audit** (`audit/AUDIT_NOTE.md`): the previous package reproduces bit-for-bit,
   but its grid-projected Gaussian BP suffers a boundary-collapse artifact at
   weakly informative t that contaminated all large-t results; and its "Gaussian
   message error" is mathematically identical to the covariance-matched Gaussian
   *model* error (equivalence proposition), which changes the interpretation.
2. **Grid BP is spectrally accurate** (exp_01): errors sit at the 1e-15 floor
   until the likelihood width ~ sqrt(2t) drops below ~2 grid steps; truncation
   matters only for A <= 4. Reference default: M = 401, A = 8.
3. **The corrected Gaussian baseline** (analytic information-form BP, exp_02/03)
   fails exactly where theory predicts: error ~ (alpha_t/Delta_t) x mean error,
   worst at small t, growing with |innovation excess kurtosis| (bimodal worst),
   and *decreasing* with stronger correlation rho.
4. **Approximate Markovianity** (exp_04): for AR(1)+global-latent priors the
   non-Markov score correction is *exactly rank one* (Woodbury); a residual MLP
   on top of the frozen Markov-BP score beats a direct score MLP by ~20x in
   sample and parameter efficiency on exact-score supervision.
5. **Reverse dynamics** (exp_05): the Gaussian score reproduces second-order
   statistics of the Laplace chain but completely washes out heavy-tailed
   innovations (excess kurtosis 0.12 vs 2.7-2.9); the score error is dynamically
   stable in L2 (trajectory divergence 0.11 despite pointwise deviation 0.49 at
   small t) yet distributionally decisive in higher moments.

## Layer 5: learning the prior (exp_06, exp_07)

Everything above hands BP the true prior. Layer 5 removes it: the transition
kernel of the clean chain becomes an unknown parameter, only *noised* sequences
are observed, and the kernel is estimated by maximum marginal likelihood with
EM. The theory is written up in `paper/main.tex` (the report this line originally pointed to,
`report/em_bp_learning.tex`, was an early draft and is now archived).

The structural points, in one place:

- **The E-step is exact.** The posterior of a chain under sitewise likelihood
  factors is a chain, chains are trees, so BP gives exact pairwise marginals.
  No variational bound, no loopy approximation.
- **The whole E-step is one matrix.** All of it compresses into an `M x M`
  matrix `Xi` — the continuum analogue of Baum-Welch's expected transition
  counts — which is sufficient and independent of how the kernel is
  parameterized. The M-step never touches the data again, and observations at
  different noise levels simply add into the same `Xi`.
- **No autodiff, anywhere.** Fisher's identity makes `<Xi, grad log K>` the
  exact gradient of the marginal log-likelihood, so BP is never
  differentiated through: it supplies the expectation. Verified against finite
  differences of the exact evidence.
- **One fit serves every noise level.** The learned kernel lives on `R x R` and
  has no `t` in it; the noise level enters only through the likelihood factors
  inside BP. A score network must instead learn a function on `R^n x R_+`
  across the whole schedule.

Four kernel rungs, increasing in expressivity, all consuming the same `Xi`:
Gaussian AR(1) and Laplace AR(1) (closed-form M-steps), a mixture-innovation
kernel that has never heard of the Laplace density it must recover, and a
mixture-density-network kernel whose gradient touches the network only at the
`M` grid points regardless of dataset size.

Two discretization artifacts are recorded rather than smoothed over, both
specific to the non-smooth Laplace kernel and absent for the smooth ones: its
`rho`-gradient loses accuracy to a `sign` discontinuity under trapezoidal
quadrature, and its exact M-step is quantized onto the grid's ratio lattice.
Reported rates therefore use the smooth kernels.

The comparison in exp_07 is set up to favour the network: the data budget is
counted in *clean* chains, of which the network gets paired `(a, x)` with a
fresh noise draw every gradient step while EM gets one noisy realization each
and never sees a clean chain; width and training budget are swept; and both
standard parameterizations (noise prediction and clean-signal prediction) are
trained and reported. The baseline is a vanilla MLP, as specified.

A locality-respecting baseline has since been measured (exp_12) and it matters:
against a weight-shared window predictor -- a 1-D CNN -- with an oracle over
both receptive-field radius and parameterization, over 4 seeds, the EM-BP margin
is 2.10 +- 0.39 / 3.27 +- 0.32 / 4.11 +- 0.32 at N = 128 / 512 / 2048, against
~10x for the vanilla MLP. So roughly 60% of the vanilla deficit was the
architecture. The margin against the CNN nevertheless *widens* with data,
because EM-BP keeps paying down parametric error while the CNN flattens toward
an approximation floor.

## Layer 6: hierarchy, speciation, and memorization (exp_13, exp_14, exp_15)

Layers 1-5 all live on a *chain*, which has exactly one correlation length. The
two papers this layer follows -- Garnier-Brun/Mezard/Moscato/Saglietti on
hierarchical filtering (arXiv:2408.15138) and Biroli/Bonnaire/de Bortoli/Mezard
on dynamical regimes (Nat. Commun. 15, 9957) -- between them supply a data model
with a *ladder* of length scales and a theory of the time scales a diffusion
passes through. `docs/PAPER_CONNECTIONS.md` is the study note; **read its section
0 first**. The transformer paper has been read in full (via a server-side route
that reaches arXiv's HTML when direct HTTP is blocked); the Nature
Communications one has **not**, and nothing mathematical is quoted from it --
the speciation crossover used here is derived from scratch and checked against
sampled trajectories. Section 0 also records two things reading the first paper
corrected: the filtering construction, which is *not* independent blocks, and a
novelty claim that Sclocchi/Favero/Wyart (PNAS 2025) already occupy.

What this adds:

- **A balanced-tree prior** with a closed-form ultrametric spectrum: exactly
  `L + 1` distinct eigenvalues, verified against `eigh` at 1e-10.
- **Exact BP on the tree**, twice: information form `(h, lambda)` matching a
  dense solve at 1e-10, and grid messages matching that at 2e-6 and working for
  any innovation law.
- **EM on the tree**, producing the *same* `Xi` as the chain E-step -- so every
  M-step in `src/kernels.py` is reused unmodified. Evidence verified against the
  closed-form Gaussian marginal likelihood at relative 1e-6, which keeps the
  monotone-ascent check available.
- **Two time scales in closed form** (`src/spectral.py`): the speciation
  crossover `t_S = 1/2 log(1 + Lambda)`, checked against 40k sampled forward
  trajectories, and the per-site excess entropy `s = -1/2 log(1 - rho^2)` that
  fixes the dataset size below which a memorizing score must collapse.

Filtering, in the paper's sense, acts on the ladder in a computable way: at
level `k` the top `k` rungs merge into one, so the number of distinct speciation
times falls from `L+1` to `L-k+2` (measured 5,5,4,3,2 at L=4). And exp_15
implements the paper's own probe -- comparison against the family of mismatched
oracles `BP_k`: a denoiser trained on data filtered at `k_train` and tested on
unfiltered data matches `BP_{k_train}` in 16 of 16 cells.

Two further results worth stating here. **A hierarchical prior shows a ladder of
speciation transitions, one per level, and the reverse diffusion resolves it
coarse-to-fine** -- six measured on a depth-5 tree, each within 3.5% of its
predicted time. And **the AR(1) chain that Layers 1-5 study has no such ladder**:
its top eigenvalue is bounded by `(1+rho)/(1-rho)` at any length, so its
speciation time saturates. That bounds how far this project generalizes toward
image-like data, and is recorded rather than glossed.

## Layout

    src/          core library (priors, noising, grid BP, Gaussian BP, exact
                  scores, Markov approximations, numpy MLP, reverse samplers,
                  EM + parameterized kernels + denoiser comparison,
                  tree priors + tree BP + tree EM, speciation/collapse scales)
    experiments/  exp_01 ... exp_27, all with --quick smoke mode
    outputs/      CSV + JSON + PNG per experiment (committed results)
    notebooks/    executed analysis notebooks 01-05
    docs/         PAPER_CONNECTIONS.md, the Layer-6 study note
    tests/        pytest suite (321 collected; 309 passed, 12 skipped locally --
                  the 12 skips are the CUDA backend-parity tests, which pass on
                  the cluster's H200 nodes)
    report/       updated_report.tex / .pdf   (Layers 1-4)
                  em_bp_learning.tex / .pdf   (Layer 5 theory)
    audit/        Layer-1 audit note

## Reproduce

Requirements: Python >= 3.12, numpy, scipy, matplotlib, pandas (analysis),
pytest (tests), jupyter/nbformat (notebooks). `pip install -r requirements.txt`.

### Start here: one command for every check

```bash
./tools/check_all.sh --quick    # ~1 min, before a commit
./tools/check_all.sh            # ~25 min, before a merge
```

Eight checks: the test suite, the provenance auditor, whether advertised paths
actually exist, whether the counts in this README match the disk, whether a
sweep is silently half-present, whether the cluster still holds both branches'
files, whether all the notebooks still re-execute, and cross-process
reproducibility.

The unglamorous ones are there deliberately. The defects that actually reached
this repository in August 2026 were not unit-testable: documentation that
outlived the runs it described, a cluster tree assembled from two branches, a
sweep split across two dated output roots by a timestamp evaluated per task.
Each check in that script carries the incident that motivated it.

To bring cluster results back and confirm what arrived:

```bash
./tools/pull_and_check.sh       # additive pull, then counts what landed
```

It reports whether the sweep is *complete*, not whether rsync exited zero — a
pull has already reported success here while transferring nothing.

### Individual pieces

```bash
python -m pytest tests/ -q                     # verify the core
python experiments/exp_01_grid_validation.py   # each writes to outputs/<name>/
python experiments/exp_02_laplace_gaussian_message_error.py
python experiments/exp_03_nongaussian_innovation_sweep.py
python experiments/exp_04_approx_markovianity.py
python experiments/exp_05_reverse_dynamics.py
python experiments/exp_06_em_parameter_recovery.py
python experiments/exp_07_em_vs_score_network.py
python experiments/exp_13_speciation_cascade.py
python experiments/exp_14_memorization_collapse.py
```

`tests/test_em_bp.py` runs many small EM fits and `tests/test_hierarchy.py` a
few tree ones; the whole suite takes **~11-12 min** (measured 2026-08-12:
309 passed, 12 skipped in 11:42 on an M-series laptop). It was ~3.7 min before
the wavelet and video suites merged in, and both this file and
`REPRODUCIBILITY.md` carried the old figure for a while. The 12 skips are the
CUDA backend-parity tests, which pass on the cluster's H200 nodes. Experiments 06, 07, 13 and 14
take roughly an hour each at full settings — use `--quick` for a minutes-scale
smoke run.

One caveat specific to the tree code: **EM on a tree needs several times more
iterations than on a chain**, because the internal nodes are never observed and
the missing-information fraction is correspondingly larger. At depth 3 with 512
trees, `rho_hat` is 0.7360 / 0.7483 / 0.7487 at 50 / 100 / 150 iterations
against a true 0.75, with zero monotone violation throughout. An estimate read
too early looks like a broken M-step; check the last-iteration change in the
log-evidence before concluding anything about accuracy.

Every experiment: deterministic seeds via `src.utils.rng_for` (common random
numbers across compared methods), full parameter dump to `params.json`,
tabular results to CSV, figures to PNG. Add `--quick` to any experiment for a
minutes-scale smoke run.

## Conventions

Forward process: dX = -X dt + sqrt(2) dW, so x_t = e^{-t} a + sqrt(1-e^{-2t}) z.
Score identity: s(x,t) = -(x - alpha_t E[a|x]) / Delta_t.
Error identity: s_hat - s_ref = (alpha_t/Delta_t)(m_hat - m_ref), verified to
machine precision in every experiment (column `identity_residual`).

## Note on reproducibility (fixed 2026-07-31)

`src/utils.rng_for` derived its seeds from Python's builtin `hash`, which is
salted per process for strings (PEP 456). Because every call mixes in a string
tag, **no experiment in this package was bit-reproducible from a fresh
interpreter**, contrary to what this README and the module docstring claimed.
The paired "common random numbers" property that the method comparisons rely on
held *within* each run, so previously reported numbers stand as measurements —
they were simply not reproducible. `rng_for` now uses a fixed digest, verified
identical across processes. Experiments 06 and 07 were run under the fix;
outputs committed for exp_01–exp_05 predate it and have not been regenerated.

## Running on a cluster

`hpc/` holds SLURM array templates and `hpc/README.md` explains them. The parts
of exp_06 and exp_07 are independent and write disjoint CSVs into a shared
output directory, so one array task per part needs no merge step:

```bash
python experiments/exp_06_em_parameter_recovery.py --list-parts
python experiments/exp_06_em_parameter_recovery.py --only rate \
    --set n_rep_rate=32 --set 'sizes_rate=(64,128,256,512,1024,4096)'
```

`--set` accepts any key shown in `params_*.json`; unknown keys are rejected so
a typo in a job script fails immediately instead of silently running defaults.

For error bars on the headline curve, `hpc/slurm_replicates.sbatch` runs the
sample-efficiency sweep under independent training seeds (`--set seed=K`) and
`tools/merge_replicates.py` merges them. The held-out test set and its exact-BP
reference are deliberately *not* reseeded, so replicates differ in what the
methods learn from and agree on what they are judged against.
