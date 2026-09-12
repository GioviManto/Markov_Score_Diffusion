# Notebooks — the theory and the practice, in the order the argument is built

Each notebook is executed and committed **with its outputs**, so what you read is what actually
ran. Where a notebook quotes a number, it re-derives that number from the committed CSVs under
`outputs/` at execution time rather than transcribing it from prose. That is deliberate: the
project has twice had documentation drift away from the data it described (see
`CLAIMS_TO_UPDATE.md`, and commits `74e6f3e` / `bd82c9b`), and a notebook that recomputes cannot
drift.

Run any of them from this directory. They add `..` to `sys.path` and read `../outputs/`.

```bash
cd notebooks
../.venv/bin/jupyter lab
```

To re-execute one end to end and confirm it still reproduces:

```bash
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 06_pointwise_vs_generative.ipynb
```

---

## The arc

**Layer 1–2 — is the numerical machinery trustworthy at all?**

| | |
|---|---|
| [`01_grid_validation.ipynb`](01_grid_validation.ipynb) | Grid BP as a calibrated numerical ground truth. Establishes that discretised belief propagation on the chain agrees with the analytic answer where one exists, and quantifies the grid error where one does not. Everything downstream is quoted against this. |

**Layer 3 — what does non-Gaussianity actually cost?**

| | |
|---|---|
| [`02_gaussian_message_error.ipynb`](02_gaussian_message_error.ipynb) | Gaussian message error in non-Gaussian chains. Sweeps innovation families — heavy tails against bimodality — and shows the error a Gaussian closure incurs is not a single number but a property of *which* way the law departs. |

**Layer 4 — locality, and the reverse process**

| | |
|---|---|
| [`03_approx_markovianity.ipynb`](03_approx_markovianity.ipynb) | Approximate Markovianity and informed scores; hybrid learning, sample and parameter efficiency. |
| [`04_reverse_dynamics.ipynb`](04_reverse_dynamics.ipynb) | Reverse diffusion dynamics — reconstruction from $t=1$, and recovery of a global non-Markov mode. |

**Layer 5 — learning the model**

| | |
|---|---|
| [`05_em_from_scratch.ipynb`](05_em_from_scratch.ipynb) | Expectation–Maximization built from nothing: BP, the E-step, the exact M-step, then the gradient route, then a mixture kernel — and finally a check of the from-scratch code against `src/`. The one to read if you want to understand the method rather than its results. |

**The results that decide whether any of it matters**

| | |
|---|---|
| [`06_pointwise_vs_generative.ipynb`](06_pointwise_vs_generative.ipynb) | **The headline.** Pointwise score accuracy does not determine generative fidelity — shown inside a single estimator, where the MLP's pointwise error improves while its generated law collapses toward a Gaussian. Includes the mechanism rebuilt in six lines, a paired check that the corrected protocol touched only the arms it was meant to, and a confound section: the *between-arm* ranking holds only at `em_components=4` and reverses at `C=8`. **Read with 09.** |
| [`07_when_the_prior_fails.ipynb`](07_when_the_prior_fails.ipynb) | **The boundary.** What happens when the Markov assumption is false. Rank-one contamination is survivable to $\beta=1.0$; long-range coupling is fatal by $\gamma\approx0.1$. Quantifies its own error bar from two zero-strength control cells, and says which conclusions survive it. |
| [`09_how_much_capacity.ipynb`](09_how_much_capacity.ipynb) | **The knob underneath 06.** Held-out evidence saturates by $C\approx8$ and cost is flat, but the fitted innovation shape is still moving at $C=16$, and the generated law is blunter than the fitted kernel at every capacity. The arm ordering flips at exactly the $C$ that likelihood-based selection picks. Ends on the `em_iters=40` confound the project's own audit note raises. |

**Beyond the chain**

| | |
|---|---|
| [`10_images_dependence_a_linear_kernel_cannot_see.ipynb`](10_images_dependence_a_linear_kernel_cannot_see.ipynb) | **First contact with real data.** CIFAR wavelet subbands are non-Gaussian by tens of standard errors, but the cross-scale dependence sits in *magnitude*, where a linear-AR kernel is structurally blind — measured against a matched linear-AR null. The flexible `mixture` kernel captures 1.4% of it and *loses* 35 nats/image against plain Gaussian; `scale_mixture` is the only one that moves the hierarchy, reaching 32%. Both conclusions sharpen when the data triples, so neither is a fitting artefact. Diagnosis confirmed, cure partial. |
| [`11_does_the_channel_destroy_shape_fastest.ipynb`](11_does_the_channel_destroy_shape_fastest.ipynb) | **A claim the project could not support, re-measured.** The original evidence compared two *second-order* quantities in a model with no shape parameter, so it could not speak to shape at all — an external review was right. With a genuine shape coordinate and nuisance-projected (efficient) information, the claim survives by 112×–1222×, two orders beyond what the old number suggested. The nuisance projection changes $\rho$'s decay fivefold and $\beta$'s barely at all. |
| [`12_how_long_does_em_need.ipynb`](12_how_long_does_em_need.ipynb) | **The confound in 09, measured — and the measurement's own limit.** 93% of seed-rows had not settled in the innovation-shape coordinate by iteration 40, the budget every `exp_16` number uses. Convergence time is set by the *design*, not $N$ or $C$. And on the `high` design the 800-update trace ran out before the answer appeared — 17 of 72 rows censored, all of them in the one coordinate the generative metrics measure. Says plainly that the capacity result is *not yet established*, and is not refuted either. |
| [`08_video_buying_time_with_space.ipynb`](08_video_buying_time_with_space.ipynb) | **The extension, and its price.** Exact BP needs a loop-free graph, so on video every temporal edge must be paid for by severing a spatial one. The held-out likelihood curve turns over — the optimum is interior — and the two axes you might optimise disagree about where it is (likelihood peaks at cut depth 2, generated coherence at depth 1). Ends with the limit stated plainly: the best model still generates motion $3.2\times$ more energetic than the reference sequences, which are CIFAR frames with synthetic rigid motion rather than video. |

---

## What is not here yet

Honest gaps, so nobody assumes coverage that does not exist:

- **`exp_24`, the Gaussian tree baseline in *pixel* coordinates**, is the one image arm still
  without a notebook — notebook 10 works in the wavelet domain and uses `exp_23`/`exp_25`. Its
  first full run completed 2026-08-12 and the results are committed under
  `outputs/exp_24_wavelet_fit/` (fit) and `exp_24_wavelet_fit_long/` (fit + denoise). Headline:
  the tree denoiser beats the raw observation by 4x at t=0.2 and 42x at t=1.5 at noise levels it
  was never fitted at, and the `mixture` kernel changes almost nothing either way — 3.5
  nats/image worse at the fitting level, ~0.5% better on denoising MSE away from it. Consistent
  with notebook 10: a flexible innovation shape does not help where the dependence is not.
- **Notebook 09's capacity numbers should be read against notebook 12**, which measures the
  convergence confound 09 names. 09 has not been rewritten in light of it: the honest status of
  the capacity result is "not established", and saying so in 12 is more useful than quietly
  restating 09.
- **The `high`-design convergence time is still unknown** — the 800-update trace censored it. A
  2400-update rerun was submitted 2026-08-12 (`623904`, `623905`); re-execute notebook 12 when
  `outputs/final_em/*/shape/*_u2400/` lands and its final cell will fill in.

## Standalone notebooks

[`standalone/`](standalone/) holds notebooks that define their own implementation
instead of importing `src/`. They are excluded from `check_all.sh` because they need
dependencies the project venv does not carry (`torch`), so **their outputs are not
verified against a fresh run** the way everything above is. See
[`standalone/README.md`](standalone/README.md).

- **`EM_BP_theory_and_experiments.ipynb`** — the full derivation plus ten experiments,
  from-scratch and self-contained. The long-form companion to notebook 05.

## Conventions

- Executed with the project venv (`../.venv`), Python 3.13, numpy 2.x, matplotlib 3.11.
- Notebooks read committed results; they do **not** re-run cluster sweeps. Small demonstrations
  built live inside a notebook (a few thousand chains) are the exception and are marked as such.
- Figures are embedded, not written to `outputs/` — `outputs/` belongs to the experiments.
- CIFAR-dependent code cannot run locally: the dataset lives only on the cluster, deliberately
  (`hpc/sync_to_cluster.sh` excludes `data/`).
