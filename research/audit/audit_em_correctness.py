"""Executable audit of the three EM correctness questions ChatGPT can only read about."""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.bp_grid import make_grid
from src.em import clean_statistics, e_step_multi, fit_clean, q_gradient, q_value
from src.kernels import GaussianAR1Kernel, LaplaceAR1Kernel, MixtureInnovationKernel
from src.noising import alpha_delta
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import rng_for

grid, w = make_grid(8.0, 301)
rho_true, n_sites, N = 0.8, 32, 128
t_vals = (0.2, 0.8)

def groups_for(prior, rng, n=N):
    A = np.stack([prior.sample(rng, n_sites) for _ in range(n)])
    out = []
    for t in t_vals:
        a, d = alpha_delta(t)
        out.append((a * A + np.sqrt(d) * rng.standard_normal(A.shape), a, d))
    return A, out

print("=" * 78)
print("TEST A  does m_step increase Q = <Xi, log K>?  (the defining M-step property)")
print("=" * 78)
lap = LaplaceAR1(rho_true)
A, grp = groups_for(lap, rng_for("audit-A"))
for name, k0 in (("Gaussian", GaussianAR1Kernel(0.2, 0.8)),
                 ("Laplace", LaplaceAR1Kernel(0.2, 0.8)),
                 ("Mixture C=8", MixtureInnovationKernel.init(8, rho=0.2, var=0.8,
                                                              rng=rng_for("audit-A-init")))):
    k = k0
    worst = np.inf
    for it in range(12):
        st = e_step_multi(grid, w, k.log_transition_matrix(grid), grp)
        q_before = q_value(st, k.log_transition_matrix(grid))
        k = k.m_step(st, grid)
        q_after = q_value(st, k.log_transition_matrix(grid))
        worst = min(worst, q_after - q_before)
    print(f"  {name:12} min Q increase over 12 M-steps: {worst:+.6e}  "
          f"{'OK' if worst >= -1e-9 else 'VIOLATION'}")

print()
print("=" * 78)
print("TEST B  is q_gradient Fisher's identity?  grad log p(x) vs central differences")
print("=" * 78)
gauss = GaussianAR1(rho_true)
_, grp_g = groups_for(gauss, rng_for("audit-B"), n=64)

def log_evidence(kernel):
    return e_step_multi(grid, w, kernel.log_transition_matrix(grid), grp_g).log_evidence

for name, k in (("Gaussian(0.75,0.40)", GaussianAR1Kernel(0.75, 0.40)),
                ("Gaussian(0.60,0.55)", GaussianAR1Kernel(0.60, 0.55))):
    st = e_step_multi(grid, w, k.log_transition_matrix(grid), grp_g)
    analytic = q_gradient(st, k.grad_log_transition_matrix(grid))
    theta = np.array(k.theta, dtype=float)
    numeric = np.zeros_like(theta)
    for j in range(len(theta)):
        h = 1e-4 * max(1.0, abs(theta[j]))
        up, dn = theta.copy(), theta.copy()
        up[j] += h; dn[j] -= h
        numeric[j] = (log_evidence(GaussianAR1Kernel(*up))
                      - log_evidence(GaussianAR1Kernel(*dn))) / (2 * h)
    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-12)
    print(f"  {name}")
    print(f"    analytic  {np.array2string(analytic, precision=6)}")
    print(f"    numeric   {np.array2string(numeric, precision=6)}")
    print(f"    max rel err {rel.max():.3e}  {'OK' if rel.max() < 5e-4 else 'MISMATCH'}")

print()
print("=" * 78)
print("TEST C  fit_clean: is holding Xi fixed correct, and is it monotone?")
print("=" * 78)
# One generator, drawn from repeatedly. Calling rng_for() inside the comprehension would reseed
# every draw and produce 512 identical chains -- which silently yields a degenerate Xi and a
# fitted kernel that looks catastrophically wrong. Cost an hour the first time.
_rng_c = rng_for("audit-C")
Ac = np.stack([lap.sample(_rng_c, n_sites) for _ in range(512)])
st_clean = clean_statistics(grid, Ac)
# Xi must not depend on the kernel at all -- clean_statistics takes no kernel.
k = MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng_for("audit-C-init"))
lls, worst = [], np.inf
for it in range(60):
    before = q_value(st_clean, k.log_transition_matrix(grid))
    k = k.m_step(st_clean, grid)
    after = q_value(st_clean, k.log_transition_matrix(grid))
    worst = min(worst, after - before)
    lls.append(after)
print(f"  clean-data Q == log-likelihood (Xi is theta-independent for clean data)")
print(f"  min increase over 60 inner M-steps: {worst:+.6e}  "
      f"{'OK' if worst >= -1e-9 else 'VIOLATION'}")
print(f"  Q at iters 1/10/30/60: {lls[0]:.4f} {lls[9]:.4f} {lls[29]:.4f} {lls[59]:.4f}")
fitted, _ = fit_clean(MixtureInnovationKernel.init(8, rho=0.3, var=0.8,
                      rng=rng_for("audit-C-init")), grid, Ac, n_iters=60)
m = fitted.innovation_moments
print(f"  fit_clean(n_iters=60) -> rho {fitted.rho:.4f} (truth {rho_true}), "
      f"var {m['innovation_var']:.4f} (truth {1-rho_true**2:.4f}), "
      f"kurt {m['innovation_excess_kurtosis']:.3f} (truth 3.0)")
