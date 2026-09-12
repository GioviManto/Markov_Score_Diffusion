"""How much does the fitted innovation kurtosis vary across DATA DRAWS at N=512?

simple/'s claim 1 reports three initialisations on ONE dataset. Those agree to four decimals,
which bounds the optimisation variance and says nothing about the estimation error. This varies
the data instead, holding everything else fixed and fitting to convergence.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.bp_grid import make_grid
from src.em import clean_statistics, e_step_multi
from src.kernels import MixtureInnovationKernel
from src.priors import LaplaceAR1
from src.protocols import one_view_groups
from src.utils import rng_for

grid, w = make_grid(8.0, 301)          # simple/'s grid
rho_true, n_sites, N, ITERS = 0.85, 32, 512, 200
t_train = (0.1, 0.2, 0.4, 0.8, 1.6)
prior = LaplaceAR1(rho_true)
print(f"truth: excess kurtosis 3.0, var {1-rho_true**2:.4f}, rho {rho_true}")
print(f"N={N}, C=8, N_g=301, {ITERS} iterations; one row per DATA draw\n")
print(f"{'draw':>5} | {'CLEAN kurt':>11} | {'NOISED kurt':>12} {'noised rho':>11}")
cl, no = [], []
for s in range(8):
    rng = rng_for("seedspread", s)
    A = np.stack([prior.sample(rng, n_sites) for _ in range(N)])
    st = clean_statistics(grid, A)
    grp = one_view_groups(A, t_train, rng_for("seedspread-g", s))
    kc = MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng_for("ss-init"))
    kn = MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng_for("ss-init"))
    for _ in range(ITERS):
        kc = kc.m_step(st, grid)
        kn = kn.m_step(e_step_multi(grid, w, kn.log_transition_matrix(grid), grp), grid)
    a = kc.innovation_moments["innovation_excess_kurtosis"]
    b = kn.innovation_moments["innovation_excess_kurtosis"]
    cl.append(a); no.append(b)
    print(f"{s:5d} | {a:11.3f} | {b:12.3f} {kn.rho:11.4f}", flush=True)
cl, no = np.array(cl), np.array(no)
for nm, v in (("clean", cl), ("noised", no)):
    print(f"\n{nm:7} mean {v.mean():.3f}  sd {v.std(ddof=1):.3f}  "
          f"se {v.std(ddof=1)/len(v)**0.5:.3f}  range [{v.min():.3f}, {v.max():.3f}]")
d = cl - no
print(f"paired clean-noised: {d.mean():+.3f} +- {d.std(ddof=1)/len(d)**0.5:.3f} "
      f"({abs(d.mean())/(d.std(ddof=1)/len(d)**0.5):.1f} sigma)")
