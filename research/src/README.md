# src — the library

Pure functions and small dataclasses. Nothing here reads a configuration file,
writes an output, or knows which experiment is running; that is `experiments/`.

## The core path

Read these four in order and you have the whole method:

| Module | What it is |
|---|---|
| `priors.py` | the clean chain — Gaussian, Laplace, mixture AR(1) generators |
| `noising.py` | the OU channel: `x = α_t a + √Δ_t z`, and the likelihood table |
| `bp_grid.py` | sum-product on the posterior chain, represented on a truncated grid |
| `em.py` | the E-step statistic Ξ, Fisher's identity, and the EM loop |
| `kernels.py` | parameterised transitions and their M-steps |

The contract `em` asks of a kernel is three methods —
`log_transition_matrix`, `grad_log_transition_matrix`, `m_step` — which is the
same interface `priors.ChainPrior` exposes. That is what lets a *learned* kernel
drop straight back into `bp_grid` and become a denoiser at every noise level
without refitting.

## What "exact" means here, precisely

Worth stating because it has been overclaimed in this codebase before:

- **Exact**: the continuous functional recursion; the analytic Gaussian
  likelihood; the E-step's structure, since the posterior really is a tree.
- **Not exact**: what actually runs. Messages are functions on ℝ represented on a
  truncated grid, so there is truncation error and O(h²) quadrature error. The
  evidence `bp_grid` returns is a quadrature estimate under that grid model.
- **Exact only sometimes**: the M-step. Closed-form maximiser for the Gaussian
  and Laplace kernels; conditional-maximisation sweeps (generalised EM) for the
  mixture and MDN.

## Everything else

`backend.py` (CPU/GPU parity), `metrics.py`, `plotting.py`, `utils.py` (seeding,
IO) are infrastructure. `ring.py`, `hierarchy.py`, `discrete.py`, `wavelet*.py`,
`video*.py`, `image_data.py` are separate models — the rotating ring is the one
that reaches the documents; the rest are exploratory branches.

## Rules that are load-bearing

- Every module is importable without side effects, and every random draw goes
  through `utils.rng_for` so a result is reproducible from its seed alone.
- Kernels are immutable; `m_step` returns a new instance.
- A run that hits its iteration cap is reported as **censored**, not returned
  looking like a converged one.

## Housekeeping

The package is pyflakes-clean: no unused imports, no assigned-and-never-used
locals, no `f"..."` without a placeholder. Worth keeping that way — three of the
findings cleared on 18 Aug 2026 were genuine leftovers of removed code, including
a CSV read in `tools/make_figures.py` that loaded a file for a panel deleted
months earlier.

```bash
./.venv/bin/python -m pyflakes src/*.py experiments/*.py tools/*.py
```
