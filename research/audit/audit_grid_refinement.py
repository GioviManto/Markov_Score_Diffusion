"""Is the residual shape deficit an information limit or a quadrature artefact?

Same chains, same initialisation, same iteration count, fitted to convergence. Only N_g varies.
If the deficit is information, it is grid-independent. If it is quadrature, it vanishes.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.bp_grid import make_grid
from src.em import clean_statistics, e_step_multi
from src.kernels import MixtureInnovationKernel
from src.priors import LaplaceAR1
from src.protocols import one_view_groups
from src.utils import rng_for

rho_true, n_sites, N, ITERS = 0.85, 32, 512, 300
t_train = (0.1, 0.2, 0.4, 0.8, 1.6)
prior = LaplaceAR1(rho_true)
rng = rng_for("grid-audit")
A = np.stack([prior.sample(rng, n_sites) for _ in range(N)])   # ONE dataset for all grids

print(f"truth: rho {rho_true}, innovation var {1-rho_true**2:.4f}, excess kurtosis 3.0")
print(f"N={N}, C=8, {ITERS} iterations, identical chains at every grid size\n")
print(f"{'N_g':>5} {'h':>7} | {'CLEAN rho':>10} {'kurt':>8} | {'NOISED rho':>11} {'kurt':>8}")
for ng in (201, 301, 401, 601, 801):
    grid, w = make_grid(8.0, ng)
    h = float(grid[1] - grid[0])
    st = clean_statistics(grid, A)
    grp = one_view_groups(A, t_train, rng_for("grid-audit-g"))
    kc = MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng_for("grid-init"))
    kn = MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng_for("grid-init"))
    for _ in range(ITERS):
        kc = kc.m_step(st, grid)
        kn = kn.m_step(e_step_multi(grid, w, kn.log_transition_matrix(grid), grp), grid)
    mc, mn = kc.innovation_moments, kn.innovation_moments
    print(f"{ng:5d} {h:7.4f} | {kc.rho:10.4f} {mc['innovation_excess_kurtosis']:8.3f} "
          f"| {kn.rho:11.4f} {mn['innovation_excess_kurtosis']:8.3f}", flush=True)
