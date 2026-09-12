# Executable audit of the EM machinery

Four scripts. They exist because reading code cannot answer these questions and running it can —
and because two of them overturned claims we had already written down.

```bash
OMP_NUM_THREADS=6 python3.12 audit/audit_em_correctness.py     # ~2 min
OMP_NUM_THREADS=6 python3.12 audit/audit_rate_vs_bias.py       # ~25 min
OMP_NUM_THREADS=8 python3.12 audit/audit_seed_spread.py        # ~35 min
OMP_NUM_THREADS=8 python3.12 audit/audit_grid_refinement.py    # ~40 min
```

## 1. `audit_em_correctness.py` — is the implementation what it claims to be?

Three properties that the code asserts in docstrings and that nothing was checking:

| check | result |
|---|---|
| every `m_step` increases `Q = <Xi, log K>` | min increase over 12 steps: **+2.61** (Gaussian), **+4.34** (Laplace), **+4.18** (mixture C=8) |
| `q_gradient` equals ∇ of the exact marginal log-likelihood | max relative error vs central differences **2.8e-7** |
| `fit_clean`'s fixed-Ξ iteration is monotone | min increase over 60 steps **+3.0e-2**, reaching ρ 0.7981 / var 0.3565 / kurt 2.822 against 0.80 / 0.36 / 3.0 |

The mixture check is the one that mattered: its `m_step` runs an *inner* EM over the mixture label
against a fixed Ξ, so it is a generalised M-step and only guaranteed to *increase* Q rather than
maximise it. It does. And `fit_clean` holding Ξ fixed across iterations is correct rather than a
shortcut, because with clean data Ξ is the empirical pair distribution and carries no dependence
on θ at all — so for clean data `Q` *is* the log-likelihood.

## 2. `audit_rate_vs_bias.py` — does the channel cost accuracy or speed?

Same chains, same initialisation, same budget; one arm sees them clean, the other once each
through the OU channel.

```
  iter |  CLEAN rho     kurt |  NOISED rho     kurt        (truth: rho 0.85, kurt 3.0)
     1 |     0.8473    1.470 |      0.4276    0.112
    30 |     0.8467    2.858 |      0.8366    1.527
   120 |     0.8473    3.014 |      0.8464    3.164
   600 |     0.8475    3.132 |      0.8466    3.096
```

**Rate, not accuracy.** Both arms reach the truth. On clean data ρ is converged after a *single*
M-step — there is no missing information at the chain level — while through the channel it takes
30–60. That is the Dempster–Laird–Rubin missing-information mechanism appearing in the convergence
rate, which is where the theory puts it.

It also separates two things we had conflated. The innovation shape is slow **even on clean data**
(tens of iterations against one for ρ), so that slowness is the mixture's inner latent label and
not the observation channel at all. The channel then slows everything further on top.

## 3. `audit_seed_spread.py` — how much does one dataset tell you?

Eight independent data draws, 200 iterations, ρ = 0.85, N = 512, C = 8:

| | mean | sd | s.e. | range |
|---|---|---|---|---|
| clean | 2.972 | 0.209 | 0.074 | [2.592, 3.293] |
| through the channel | 3.024 | 0.577 | 0.204 | [2.396, 3.993] |

paired clean − noised: **−0.052 ± 0.217**, i.e. no channel penalty on shape at convergence.

**This retired a claim.** `SIMPLE_RESULTS.md` reported "shape is recovered to about 72%" and built
an argument on the missing 28%, weighing mixture capacity against channel information. That 2.15
was one draw at an iteration count it had not converged at; single draws span 2.40 to 3.99. Note
also that the three initialisations reported in claim 1 share one dataset, so their agreement
bounds optimisation variance and not estimation error — a distinction the original text blurred.

## 4. `audit_grid_refinement.py` — is any of it quadrature?

Identical chains at N_g ∈ {201, 301, 401}, fitted to convergence: clean kurtosis 3.461 / 3.485 /
3.494, channel-fitted 3.528 at all three. Grid-independent to about 1%, so the shape estimate is
not quadrature-limited at these sizes and the deficits above were never a grid artefact.

## The general lesson

**A fixed iteration budget compares convergence rates, not estimators.** Whatever converges slower
looks worse, so under-iterated runs produce apparent *deficits* rather than symmetric noise — which
reads convincingly as a real effect. It has cost this project three separate false findings:

- `exp_16` at `em_iters=40`: a pointwise/generative dissociation plus a "capacity effect", of which
  roughly 96% was convergence rate at a fixed budget.
- `simple/` at 120: the 28% shape deficit above.
- `exp_06`'s clean-vs-noised at a fixed 120: a channel penalty of +0.22 ± 0.15 that is
  −0.05 ± 0.22 once both arms are run to convergence.

Prefer a convergence criterion to a fixed count, and when a fixed count is unavoidable, make it
generous and check the slowest coordinate rather than the fastest.
