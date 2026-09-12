"""Regression tests for the September 2026 scientific-audit corrections.

Each test pins one repaired claim so that a regeneration or refactor cannot
silently reintroduce the defect it fixed. The thesis locations are noted so a
failure points at prose as well as code.
"""
import csv
import os

import numpy as np
import pytest

RHO = 0.85
L = 32
SCHEDULE = (0.05, 0.0725, 0.1053, 0.1527, 0.2216, 0.3215,
            0.4665, 0.6769, 0.9821, 1.4250, 2.0676, 3.0)


def ar1_cov(rho=RHO, n=L):
    i = np.arange(n)
    return rho ** np.abs(i[:, None] - i[None, :])


# ---------------------------------------------------------------------------
# M-01: the pooled estimand is not the mean of per-level errors or ratios.
# ---------------------------------------------------------------------------
def pooled_rel_l2(errs, refs):
    """sqrt(sum_t B_t E_t^2 / sum_t B_t): thesis Eq. (10.2) given per-level
    relative errors E_t and reference energies B_t."""
    errs, refs = np.asarray(errs, float), np.asarray(refs, float)
    return float(np.sqrt((refs * errs**2).sum() / refs.sum()))


def test_pooled_estimand_differs_from_mean_of_errors():
    # The audit's synthetic example: a small error on a high-energy level and
    # a large error on a negligible-energy level. The arithmetic mean is 0.5;
    # the pooled value is close to the high-energy level's error.
    e = pooled_rel_l2([0.1, 0.9], [100.0, 1.0])
    assert abs(np.mean([0.1, 0.9]) - 0.5) < 1e-15
    assert abs(e - np.sqrt((100 * 0.01 + 1 * 0.81) / 101.0)) < 1e-15
    assert e < 0.14  # nowhere near 0.5: a generator that silently switches fails


def test_pooled_paired_ratio_differs_from_mean_of_ratios():
    # Deterministic, shaped like the real data: per-level ratios grow with t
    # while the reference energy falls, so the mean of ratios up-weights
    # exactly the levels the pooled estimand down-weights.
    k = len(SCHEDULE)
    refs = np.geomspace(100.0, 1.0, k)
    e_em = np.full(k, 0.05)
    e_net = e_em * np.linspace(2.0, 30.0, k)
    pooled = pooled_rel_l2(e_net, refs) / pooled_rel_l2(e_em, refs)
    mean_of_ratios = float(np.mean(e_net / e_em))
    assert mean_of_ratios > 1.5 * pooled


def test_frozen_reference_energies_match_headline_table():
    """The recovered B_t reproduce the shipped headline range 6.6-10.9."""
    # Relative to THIS FILE, not the working directory: as a cwd-relative
    # path this skipped silently whenever pytest was run from anywhere but
    # the package root -- including the repository root of the published
    # copy, which is the natural place to run it.
    from pathlib import Path as _P
    ref_csv = _P(__file__).resolve().parents[1] / "outputs" / "frozen" \
        / "exp_07_ref_energy" / "ref_energy.csv"
    if not ref_csv.exists():
        pytest.skip(f"reference energies not generated at {ref_csv}")
    B = {float(r["t"]): float(r["ref_sq"]) for r in csv.DictReader(open(ref_csv))}
    assert set(np.round(list(B), 4)) == set(np.round(SCHEDULE, 4))
    # energies must be positive and decreasing towards the terminal levels
    v = [B[t] for t in SCHEDULE]
    assert all(x > 0 for x in v)
    assert v[0] > 3 * v[-1]


# ---------------------------------------------------------------------------
# M-02: tr(Sigma_t^{-1}) has a finite t->0 limit; it is NOT L/(2t).
# Thesis Eq. (10.3) region.
# ---------------------------------------------------------------------------
def test_score_energy_low_noise_limit():
    S0 = ar1_cov()
    Q0 = np.linalg.inv(S0)
    trQ0 = np.trace(Q0)
    coeff = 2.0 * (np.trace(Q0) - np.trace(Q0 @ Q0))
    for t in (1e-3, 1e-4, 1e-5):
        St = np.exp(-2 * t) * S0 + (1 - np.exp(-2 * t)) * np.eye(L)
        tr = np.trace(np.linalg.inv(St))
        # finite limit with the first-order expansion
        assert abs(tr - (trQ0 + coeff * t)) < 1e5 * t**2 + 1e-8
        # and explicitly NOT L/(2t)
        assert abs(tr - L / (2 * t)) > 0.5 * L / (2 * t)


# ---------------------------------------------------------------------------
# Ch5: band-fill leading order and all-t off-band density (alpha != 0).
# ---------------------------------------------------------------------------
def _Qt(rho, t, n=L):
    S0 = ar1_cov(rho, n)
    St = np.exp(-2 * t) * S0 + (1 - np.exp(-2 * t)) * np.eye(n)
    return np.linalg.inv(St)


def test_band_fill_leading_order():
    S0 = ar1_cov()
    Q0 = np.linalg.inv(S0)
    k = L // 2
    for d in (2, 3):
        Qd = np.linalg.matrix_power(Q0, d)
        ts = np.array([4e-4, 2e-4, 1e-4])
        vals = np.array([_Qt(RHO, t)[k, k + d] for t in ts])
        pred = np.array([(-1) ** (d - 1) * (2 * t) ** (d - 1) * Qd[k, k + d]
                         for t in ts])
        assert np.all(np.abs(vals / pred - 1.0) < 0.05)


def test_offband_density_and_signs():
    rng = np.random.default_rng(1)
    for rho in (0.85, 0.3, -0.6):
        for t in (0.01, 0.4, 2.0):
            Q = _Qt(rho, t, 12)
            off = Q[~np.eye(12, dtype=bool)]
            assert np.all(np.abs(off) > 0), (rho, t)
            if rho > 0:
                assert np.all(off < 0), "positive alpha: attractive couplings"
    # alpha = 0: exactly the identity under variance preservation
    Q = _Qt(0.0, 0.7, 12)
    assert np.allclose(Q, np.eye(12), atol=1e-12)


# ---------------------------------------------------------------------------
# M-06: the exact windowed-oracle risk, sanity anchors.
# ---------------------------------------------------------------------------
def _windowed_matrix(rho, t, r, n=L):
    S0 = ar1_cov(rho, n)
    al, de = np.exp(-t), 1 - np.exp(-2 * t)
    M = np.zeros((n, n))
    for k in range(n):
        lo, hi = max(0, k - r), min(n, k + r + 1)
        Jw = np.linalg.inv(S0[lo:hi, lo:hi]) + (al * al / de) * np.eye(hi - lo)
        M[k, lo:hi] = (al / de) * np.linalg.solve(Jw, np.eye(hi - lo)[:, k - lo])
    return (al * M - np.eye(n)) / de


def test_windowed_oracle_full_window_is_exact():
    t = 0.3
    Shat = _windowed_matrix(RHO, t, L)          # window covers the whole chain
    Qt = _Qt(RHO, t)
    assert np.allclose(Shat, -Qt, atol=1e-9)


def test_windowed_oracle_decreasing_in_radius():
    t = 0.3
    S0 = ar1_cov()
    St = np.exp(-2 * t) * S0 + (1 - np.exp(-2 * t)) * np.eye(L)
    Qt = np.linalg.inv(St)

    def err(r):
        D = _windowed_matrix(RHO, t, r) + Qt
        return np.sqrt(np.trace(D @ St @ D.T) / np.trace(Qt))
    e = [err(r) for r in (0, 1, 2, 4, 8)]
    assert all(a > b for a, b in zip(e, e[1:]))
    assert e[0] > 0.2      # r=0 is far from exact even though t is moderate


# ---------------------------------------------------------------------------
# Ch7: the chain-state fourth cumulant (not the innovation's 3).
# ---------------------------------------------------------------------------
def test_laplace_state_kurtosis():
    rho = RHO
    k4_pred = 3.0 * (1 - rho**2) / (1 + rho**2)      # bulk limit, unit variance
    rng = np.random.default_rng(2)
    n, reps = 400, 200_000
    b = np.sqrt((1 - rho**2) / 2.0)
    a = rng.standard_normal(reps)
    for _ in range(n):
        a = rho * a + rng.laplace(0.0, b, size=reps)
    m2 = np.mean(a**2)
    k4 = np.mean(a**4) - 3 * m2**2
    assert abs(m2 - 1.0) < 0.02
    assert abs(k4 - k4_pred) < 0.06                   # ~0.483 at rho=0.85
    assert k4 < 1.0, "state kurtosis is far below the innovation's 3"


# ---------------------------------------------------------------------------
# Ch10: window-head parameter count matches the (2r+4)W + W^2 + 3W + 1 formula.
# ---------------------------------------------------------------------------
def test_window_head_parameter_count():
    from src.nnet import MLP
    rng = np.random.default_rng(3)
    for r, expect in ((2, 4801), (4, 5057)):
        net = MLP.init((2 * r + 1 + 3, 64, 64, 1), rng)
        formula = (2 * r + 4) * 64 + 64 * 64 + 3 * 64 + 1
        assert net.n_params == formula == expect


# ---------------------------------------------------------------------------
# M-03: the trace/log-det scalar is the Gaussian KL (checked by Monte Carlo).
# ---------------------------------------------------------------------------
def test_delta_is_gaussian_kl():
    rng = np.random.default_rng(4)
    n = 4
    A = rng.standard_normal((n, n)); Sp = A @ A.T + n * np.eye(n)
    Bm = rng.standard_normal((n, n)); Sq = Bm @ Bm.T + n * np.eye(n)
    formula = 0.5 * (np.trace(np.linalg.solve(Sq, Sp)) - n
                     + np.linalg.slogdet(Sq)[1] - np.linalg.slogdet(Sp)[1])
    X = rng.multivariate_normal(np.zeros(n), Sp, size=400_000)
    Qp, Qq = np.linalg.inv(Sp), np.linalg.inv(Sq)
    logp = -0.5 * np.einsum("ij,jk,ik->i", X, Qp, X) - 0.5 * np.linalg.slogdet(Sp)[1]
    logq = -0.5 * np.einsum("ij,jk,ik->i", X, Qq, X) - 0.5 * np.linalg.slogdet(Sq)[1]
    assert abs(np.mean(logp - logq) - formula) < 0.01


# ---------------------------------------------------------------------------
# Ch7: composite trapezoidal weights really carry the halved endpoints.
# ---------------------------------------------------------------------------
def test_trapezoid_endpoint_weights():
    from src.bp_grid import make_grid
    grid, w = make_grid(8.0, 101)
    h = grid[1] - grid[0]
    assert np.isclose(w[0], h / 2) and np.isclose(w[-1], h / 2)
    assert np.allclose(w[1:-1], h)
    assert np.isclose(w.sum(), 16.0)


# ---------------------------------------------------------------------------
# Ch9: the Dempster Jacobian belongs to EXACT EM, not to the ECM map that is
# actually run. In the quadratic idealisation with r inner conditional-
# maximisation sweeps the implemented map's Jacobian is
#     DM_ECM(r) = G^r + (I - G^r) DM_EM,
# G being the block Gauss-Seidel matrix of the CM steps, so it agrees with
# I_com^{-1} I_mis only in the limit of a fully maximising M-step. Chapter 9
# states (9.2) as the exact-EM benchmark for this reason; the test pins all
# three facts so the distinction cannot quietly be dropped again.
# ---------------------------------------------------------------------------
def test_ecm_jacobian_differs_from_missing_information():
    rng = np.random.default_rng(11)
    n = 3
    # A valid (I_com, I_mis) pair: both symmetric PD, I_mis strictly smaller.
    R = rng.standard_normal((n, n))
    I_com = R @ R.T + n * np.eye(n)
    S = rng.standard_normal((n, n))
    I_mis = 0.35 * (S @ S.T + np.eye(n))
    I_obs = I_com - I_mis
    assert np.all(np.linalg.eigvalsh(I_obs) > 0), "I_obs must stay PD"

    # Exact EM on the quadratic: argmax_theta Q(theta|phi) = m(phi), so DM_EM
    # is exactly Dempster's I_com^{-1} I_mis.
    DM_EM = np.linalg.solve(I_com, I_mis)
    star = rng.standard_normal(n)

    def m(phi):
        return star + DM_EM @ (phi - star)

    def ecm_step(phi, r):
        """r sweeps of EXACT coordinatewise conditional maximisation of
        Q(theta|phi) = -1/2 (theta-m(phi))^T I_com (theta-m(phi)), started
        from phi. Each coordinate is solved in closed form with the others
        held fixed -- the real algorithm, not the formula under test."""
        target, th = m(phi), phi.copy()
        for _ in range(r):
            for j in range(n):
                # d/dtheta_j of Q = 0  =>  theta_j = target_j - (sum_{k!=j}
                # I_com[j,k](theta_k - target_k)) / I_com[j,j]
                off = I_com[j] @ (th - target) - I_com[j, j] * (th[j] - target[j])
                th[j] = target[j] - off / I_com[j, j]
        return th

    def jacobian(f, at, eps=1e-6):
        cols = []
        for j in range(n):
            e = np.zeros(n); e[j] = eps
            cols.append((f(at + e) - f(at - e)) / (2 * eps))
        return np.stack(cols, axis=1)

    L = np.tril(I_com)
    G = np.eye(n) - np.linalg.solve(L, I_com)

    def predicted(r):
        Gr = np.linalg.matrix_power(G, r)
        return Gr + (np.eye(n) - Gr) @ DM_EM

    # (i) The claimed formula is the Jacobian of the ACTUAL ECM map, checked by
    # numerical differentiation rather than by restating the formula. This is
    # what rejects a dropped G^r term, which a formula-vs-formula test cannot.
    for r in (1, 2, 4):
        measured = jacobian(lambda p_: ecm_step(p_, r), star)
        assert np.allclose(measured, predicted(r), atol=1e-6), (r, measured)
        # The dropped-G^r variant is only distinguishable while G^r is not yet
        # negligible: G is a contraction, so both forms tend to DM_EM as the
        # M-step is solved more exactly. Assert the rejection where it has
        # discriminating power, which is exactly the regime the thesis claims
        # matters (a capped inner loop).
        Gr = np.linalg.matrix_power(G, r)
        if np.linalg.norm(Gr) > 1e-2:
            wrong = (np.eye(n) - Gr) @ DM_EM
            assert not np.allclose(measured, wrong, atol=1e-3), r

    # (ii) One finite sweep is not the Dempster Jacobian, in matrix and in
    # spectral radius, so a measured contraction is not missing information.
    one = jacobian(lambda p_: ecm_step(p_, 1), star)
    assert not np.allclose(one, DM_EM, atol=1e-3)
    rho = lambda M_: float(np.max(np.abs(np.linalg.eigvals(M_))))
    assert abs(rho(one) - rho(DM_EM)) > 1e-3

    # (iii) ... and the gap closes as the M-step is solved more exactly.
    gaps = [np.linalg.norm(jacobian(lambda p_, r=r: ecm_step(p_, r), star) - DM_EM)
            for r in (1, 2, 4, 8, 16)]
    assert all(b < a for a, b in zip(gaps, gaps[1:])), gaps


# ---------------------------------------------------------------------------
# Ch10: the measured Laplace local oracle (thesis/tools/make_laplace_oracle.py).
# The strongest available correctness check is the degenerate one: when the
# window is the whole chain the "oracle" IS the exact score, so the relative
# error must vanish identically. A window prior that refreshed N(0,1) instead
# of carrying the chain's true state marginal, a mismatched pooling, or a sign
# slip in the Tweedie conversion would all leave a non-zero floor here.
# ---------------------------------------------------------------------------
def test_laplace_oracle_vanishes_on_full_window():
    import sys
    from pathlib import Path
    here = Path(__file__).resolve()
    # Two layouts hold this suite: the working tree, where the documents sit
    # beside the research package, and the published repository, where the
    # thesis is at the root and the research package is a subdirectory. Look
    # for the generator in both -- a path that matches only one of them makes
    # this test skip silently on the other, which is exactly what it did in the
    # published copy until 12 Sep 2026.
    candidates = [
        here.parents[3] / "Markov_Score_Diffusion" / "thesis" / "tools",
        here.parents[2] / "tools",
    ]
    tools = next((c for c in candidates if (c / "make_laplace_oracle.py").exists()), None)
    if tools is None:                           # research tree checked out alone
        import pytest
        pytest.skip("make_laplace_oracle.py not found in either layout: "
                    + ", ".join(str(c) for c in candidates))
    sys.path.insert(0, str(tools))
    import make_laplace_oracle as M

    M.SCHEDULE = [0.05, 0.4]                    # two levels: this is a shape test
    M.RADII = [1, M.NSITES - 1]
    point, _ = M.laplace_oracle(8, seed=3)

    assert point[M.NSITES - 1] < 1e-12, point[M.NSITES - 1]
    # ... and a genuinely local window is NOT free, so the test cannot pass
    # by the whole computation collapsing to zero.
    assert point[1] > 1e-2, point[1]
