"""Separate the RATE of shape convergence from its ASYMPTOTE.

Same chains, same kernel, same initialisation, same iteration budget. One arm sees them clean
(no channel, Xi exact and theta-independent); the other sees each chain once through the OU
channel at one of five levels. If the channel caused the slowness, the clean arm would be fast.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.bp_grid import make_grid
from src.em import clean_statistics, e_step_multi
from src.kernels import MixtureInnovationKernel
from src.priors import LaplaceAR1
from src.protocols import one_view_groups
from src.utils import rng_for

grid, w = make_grid(8.0, 401)
rho_true, n_sites, N = 0.85, 32, 512
t_train = (0.1, 0.2, 0.4, 0.8, 1.6)
prior = LaplaceAR1(rho_true)
rng = rng_for("rate-vs-bias")
A = np.stack([prior.sample(rng, n_sites) for _ in range(N)])

st_clean = clean_statistics(grid, A)
grp = one_view_groups(A, t_train, rng_for("rate-vs-bias-g"))

def init():
    return MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng_for("rate-init"))

marks = (1, 5, 10, 20, 30, 60, 120, 240, 400, 600)
print(f"truth: rho {rho_true}, innovation var {1-rho_true**2:.4f}, excess kurtosis 3.0")
print(f"{'iter':>6} | {'CLEAN rho':>10} {'kurt':>8} | {'NOISED rho':>11} {'kurt':>8}")
kc, kn = init(), init()
for it in range(1, max(marks) + 1):
    kc = kc.m_step(st_clean, grid)                                    # Xi fixed: no channel
    kn = kn.m_step(e_step_multi(grid, w, kn.log_transition_matrix(grid), grp), grid)
    if it in marks:
        mc, mn = kc.innovation_moments, kn.innovation_moments
        print(f"{it:6d} | {kc.rho:10.4f} {mc['innovation_excess_kurtosis']:8.3f} "
              f"| {kn.rho:11.4f} {mn['innovation_excess_kurtosis']:8.3f}", flush=True)
