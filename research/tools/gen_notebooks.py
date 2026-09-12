"""Generate the four analysis notebooks (nbformat), to be executed afterwards."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

SETUP = """\
from pathlib import Path
import pandas as pd
from IPython.display import Image, display, Markdown

OUT = Path("..") / "outputs"
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
def show(*names, exp):
    for n in names:
        display(Image(filename=str(OUT / exp / n)))
"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    return n


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


# ---------------------------------------------------------------- notebook 1
nb1 = nb([
    md("""\
# 01 — Grid BP as a calibrated numerical ground truth (Layer 2)

For a Gaussian AR(1) prior the exact score is available by linear algebra:

$$\\Sigma_0(i,j)=\\rho^{|i-j|},\\qquad \\Sigma_t=\\alpha_t^2\\Sigma_0+\\Delta_t I,\\qquad
s_{\\rm true}(x,t)=-\\Sigma_t^{-1}x .$$

Grid BP represents the (mathematically exact) continuous BP messages on an equally
spaced grid on $[-A, A]$ with trapezoidal quadrature. This notebook quantifies its two
error sources — tail truncation ($A$) and resolution ($M$) — against the exact score,
using **common random numbers** across every grid configuration.

Rerun: `python experiments/exp_01_grid_validation.py` (add `--quick` for a smoke run).\
"""),
    code(SETUP),
    code("""\
exp = "exp_01_grid_validation"
sweep_m = pd.read_csv(OUT / exp / "grid_sweep_M.csv")
sweep_m[(sweep_m.rho == 0.95) & (sweep_m.t <= 0.05)][
    ["grid_size", "t", "score_rel_error_mean", "resolution_ok"]
]"""),
    md("""\
**Key observation — spectral accuracy.** The trapezoid rule on analytic, rapidly
decaying integrands (Gaussian kernels/likelihoods) converges *exponentially* in the
number of points per likelihood width, not at the naive $O(dx^2)$ rate. Consequently
grid BP sits at the $10^{-15}$ floor almost everywhere, and failure is *abrupt*: it
happens only when the likelihood width $\\sqrt{\\Delta_t}/\\alpha_t \\approx \\sqrt{2t}$
falls below ~2 grid steps (top-left of the table: $M=101$, $t=0.005$). Truncation error
appears only at $A=4$ (~$5\\times10^{-7}$) and is already negligible at $A=5$ for
Gaussian tails."""),
    code("""show("grid_error_vs_t.png", "grid_error_vs_M.png", "grid_error_vs_A.png", exp=exp)"""),
    code("""show("grid_heatmap_t0.005.png", "grid_heatmap_t0.4.png", exp=exp)"""),
    code("""print((OUT / exp / "RECOMMENDATION.md").read_text())"""),
])

# ---------------------------------------------------------------- notebook 2
nb2 = nb([
    md("""\
# 02 — Gaussian message error in non-Gaussian Markov chains (Layer 3)

**Equivalence proposition (message ≡ model at the single-Gaussian level).** For a
linear-transition chain $a_i=\\rho a_{i-1}+\\varepsilon_i$ with $\\mathrm{Var}(\\varepsilon)=q$
and Gaussian OU likelihoods, moment-matched single-Gaussian (assumed-density) BP is
*identical* to exact Gaussian BP under the covariance-matched Gaussian AR(1) prior:
the tilted product (Gaussian message × Gaussian likelihood) is exactly Gaussian, and the
transition step maps $N(m,v)$ to a density whose first two moments $(\\rho m,\\ \\rho^2v+q)$
do not depend on the innovation shape. So "Gaussian message error" and "Gaussian model
error" coincide here; separating them requires richer message families (e.g. mixtures).

The *analytic information-form* implementation of this baseline (`gaussian_chain_bp`)
replaces the legacy grid-mediated projection, whose backward messages collapse onto the
grid boundary at weakly informative $t$ (audit finding F1).

Rerun: `python experiments/exp_02_laplace_gaussian_message_error.py` and
`python experiments/exp_03_nongaussian_innovation_sweep.py`.\
"""),
    code(SETUP),
    code("""\
exp2 = "exp_02_laplace_gaussian_message_error"
summ = pd.read_csv(OUT / exp2 / "laplace_summary.csv")
summ[["t", "rel_score_error_median", "rel_score_error_q90", "p_rel_error_gt_0p1",
      "grid_self_conv_median", "legacy_artifact_max"]]"""),
    md("""\
Three things to read off this table:
1. the corrected closure error decays **monotonically** with $t$ — the old package's
   large-$t$ anomalies were the legacy artifact (last column: posterior-mean errors up
   to ~6.6 at $t\\geq0.9$, in a regime where the true closure error is $<10^{-2}$);
2. the grid self-convergence error stays 4–7 orders of magnitude below the closure
   error, so the reference is trustworthy;
3. at small $t$ the Gaussian baseline fails for *every* trial
   ($P(\\mathrm{err}>0.1)=1$), as predicted by the $\\alpha_t/\\Delta_t$ amplification."""),
    code("""show("laplace_closure_vs_grid_budget.png", "legacy_artifact_size.png",
     "laplace_worst_case_belief.png", exp=exp2)"""),
    md("## Innovation sweep: heavy tails vs bimodality"),
    code("""\
exp3 = "exp_03_nongaussian_innovation_sweep"
print((OUT / exp3 / "SUMMARY_TABLE.md").read_text())"""),
    code("""show("sweep_error_vs_t.png", "sweep_error_vs_rho.png", "sweep_error_vs_kurtosis.png", exp=exp3)"""),
    md("""\
Findings: error grows with $|$excess kurtosis$|$ in both directions (bimodal mixtures
are the worst case at equal $|\\kappa_4|$); **stronger correlation $\\rho$ *helps* the
Gaussian baseline** at fixed $t$ (informative neighbours Gaussianize the local
posterior), contrary to the naive error-propagation intuition."""),
])

# ---------------------------------------------------------------- notebook 3
nb3 = nb([
    md("""\
# 03 — Approximate Markovianity and informed scores (Layer 4)

Gaussian non-Markov models, exact linear algebra everywhere.

- **Model A**: $a=(y+\\beta g)/\\sqrt{1+\\beta^2}$ with $y$ AR(1), $g$ a global latent.
  For the naive chain approximation the score residual is **exactly rank one** at every
  $t$ (Woodbury): $s_{\\rm true}-s_{\\rm chain} = c_t(x)\\, \\Sigma_{t}^{-1}\\mathbf{1}$ —
  a fixed smooth direction with an $x$-linear coefficient.
- **Model B**: precision $Q = Q_{\\rm AR1}+\\gamma Q_{\\rm long}$ — a genuinely
  higher-rank correction.
- **Hybrid learning**: direct MLP score vs (frozen) Markov-BP score + MLP *residual*,
  trained on exact-score targets.

Rerun: `python experiments/exp_04_approx_markovianity.py`.\
"""),
    code(SETUP),
    code("""\
exp = "exp_04_approx_markovianity"
res = pd.read_csv(OUT / exp / "residual_operator_analysis.csv")
res[(res.model == "A_beta") & (res.t == 0.2)][
    ["strength", "approx", "rel_residual_median", "sv2_over_sv1", "effective_rank"]
]"""),
    md("""\
The naive chain's residual operator has $\\sigma_2/\\sigma_1 \\sim 10^{-12}$: *exactly*
rank one. The Chow-Liu chain has smaller residual *norm* but spreads it over more
directions — structure and norm trade off."""),
    code("""show("modelA_residual_norm.png", "modelA_rank_one_dominance.png",
     "modelB_residual_norm.png", "modelB_effective_rank.png", exp=exp)"""),
    md("## Hybrid learning: sample and parameter efficiency"),
    code("""\
hyb = pd.read_csv(OUT / exp / "hybrid_learning.csv")
hyb"""),
    code("""show("hybrid_sample_efficiency.png", "hybrid_parameter_efficiency.png", exp=exp)"""),
    md("""\
The residual learner beats the direct score net by an order of magnitude at every
training-set size and parameter count, and beats the no-learning Markov baseline from
256 samples on. The direct net does not reach the *free* Markov baseline even with
4096 samples. Caveat: the Markov model is known here, not estimated; this isolates the
value of the structural prior, not of the estimation pipeline."""),
])

# ---------------------------------------------------------------- notebook 4
nb4 = nb([
    md("""\
# 04 — Reverse diffusion dynamics (Layer 4 / dynamics)

Convention: forward $dX=-X\\,dt+\\sqrt2\\,dW$; reverse Euler-Maruyama
$x \\leftarrow x + h(x+2s) + \\sqrt{2h}\\,\\xi$ on a geometric time grid from $T=3$
to $t_{\\min}=0.01$; probability-flow ODE $\\dot x = -x - s$ with Heun steps.
Paired runs share initial noise and Brownian increments.

Rerun: `python experiments/exp_05_reverse_dynamics.py`.\
"""),
    code(SETUP),
    code("""\
exp = "exp_05_reverse_dynamics"
pd.read_csv(OUT / exp / "setup1_terminal_stats.csv")"""),
    md("""\
**Does the Gaussian score wash out non-Gaussian structure?** Compare the excess
kurtosis of fitted innovations $\\hat\\varepsilon_i = a_i - \\rho a_{i-1}$ in reverse
samples against clean prior samples (Laplace innovations have excess kurtosis $+3$)."""),
    code("""show("setup1_innovation_kurtosis.png", "setup1_pointwise_vs_dynamic.png", exp=exp)"""),
    code("""\
div = pd.read_csv(OUT / exp / "setup1_divergence.csv")
div.tail(8)"""),
    md("## Reconstruction from t = 1"),
    code("""show("setup2_correlation_recovery.png", exp=exp)"""),
    md("## Model A: recovery of the global (non-Markov) mode"),
    code("""\
pd.read_csv(OUT / exp / "setup3_modelA_terminal.csv")"""),
    code("""show("setup3_global_mode_recovery.png", exp=exp)"""),
])

for name, notebook in [
    ("01_grid_validation.ipynb", nb1),
    ("02_gaussian_message_error.ipynb", nb2),
    ("03_approx_markovianity.ipynb", nb3),
    ("04_reverse_dynamics.ipynb", nb4),
]:
    nbf.write(notebook, NB_DIR / name)
    print("wrote", NB_DIR / name)
