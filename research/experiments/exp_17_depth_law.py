"""Experiment 17 -- how deep must a network be to run BP on a chain?

The question, in Jerome's words
-------------------------------
This was his stated "biggest question at the moment" in the call of 29 July:

    "From a neural network perspective, it's a bit weird because a chain, the most natural
     way to run BP on a chain would be a neural network which is as deep as twice the chain
     ... but has a single neuron. [...] So for me, the biggest question at the moment is,
     can this BP of a very simplistic tree -- what is the required depth of a neural network
     that you need to learn it?"

And he contrasted it with the tree case, where the answer is known: depth equals tree depth,
because BP must go up and back down. A chain is a tree, but a degenerate one -- going "up"
advances a single node -- so the depth requirement should scale with the *chain length*
rather than with a log of it, while the width requirement stays trivial. He sketched a
sparse, narrow, deep architecture on the call and asked whether it is really necessary.

Why this is answerable exactly, rather than by a sweep
------------------------------------------------------
There is a quantitative prediction available, which is what makes this a law to test rather
than a hyperparameter search.

Information reaches site ``i`` from site ``j`` only after ``|i - j|`` message-passing steps.
A network whose layers mix only neighbouring sites therefore has an information horizon
equal to its depth: a depth-``d`` network can, at absolute best, implement the estimator that
sees a window of radius ``d``. And for the Gaussian chain the project already knows that
estimator's error **in closed form** -- ledger item G12, audited to 0.2% on the slope:

    RMS error of the radius-r estimator decays exactly as  q^r,
        q = (J_d - sqrt(J_d^2 - 4 beta^2)) / (2 |beta|)

So the falsifiable statement is:

    **error(depth d)  >=  error(radius d)  ~  q^d**

with equality if the network is efficient at using its horizon. That gives three outcomes,
all informative:

  * the network tracks ``q^d`` -- depth *is* the binding constraint, and BP's message count
    is the right way to think about architecture;
  * the network plateaus above ``q^d`` -- depth is available but not exploited, i.e. the
    optimisation, not the architecture, is the limit;
  * the network beats ``q^d`` -- the horizon argument is wrong and something is leaking
    information across sites faster than the layers should allow (in practice: a bug, most
    likely a non-local layer or padding that sees the whole sequence).

The third is a genuine correctness check, not a hypothetical: it is exactly how one would
catch an architecture that is accidentally global.

Why the Gaussian chain
----------------------
Deliberately the *easy* case, and this is a scope limitation not a hidden assumption. For a
Gaussian AR(1) the radius-r estimator is the optimal linear predictor from a window, so it is
computable in closed form with no training and no grid -- which means the reference curve
carries zero estimation error and any gap is entirely the network's. On a non-Gaussian chain
the radius-r reference would itself have to be estimated, and a gap could not be attributed.
The law is about *information flow*, which is a property of the graph rather than of the
innovation law, so testing it where the reference is exact is the right order of business.
Whether the constant changes for non-Gaussian innovations is a separate question, and exp_11
already found locality rates there are up to 2.45x the Gaussian ones.

Parts
-----
law     Verify the reference itself: does the exact radius-r estimator decay as q^r, with
        the fitted slope matching the closed form? Nothing downstream is meaningful if not.
depth   Train chain-local networks of depth d = 1..D and compare to radius-d.
budget  The control that makes the answer about *depth*. Hold parameters fixed and vary the
        depth/width trade; if error tracked parameter count rather than depth, the headline
        would be vacuous.
"""

from __future__ import annotations

import numpy as np

from common import apply_overrides, experiment_parser, provenance
from src.exact_scores import exact_gaussian_posterior_mean, sigma_t
from src.spectral import chain_covariance
from src.utils import ensure_dir, rng_for, write_csv, write_json

SETTINGS = {
    "n_sites": 32,
    "rho": 0.85,
    "t_probe": (0.1, 0.2, 0.4, 0.8, 1.6),
    "radii": tuple(range(0, 13)),
    "depths": (1, 2, 3, 4, 6, 8, 10, 12),
    "n_train": 4096,
    "n_test": 2048,
    "width": 24,
    "n_steps": 4000,
    "batch_size": 128,
    "lr": 2e-3,
    "n_seed": 3,
    "seed_offset": 0,
    # For the budget control: (depth, width) pairs chosen to hold parameter count roughly
    # constant, so depth varies at fixed capacity.
    "budget_pairs": ((2, 58), (4, 40), (6, 32), (8, 28), (12, 22)),
}

QUICK = {
    "n_train": 512, "n_test": 512, "n_steps": 300, "n_seed": 1,
    "depths": (1, 2, 4), "radii": (0, 1, 2, 4, 8), "t_probe": (0.2, 0.8),
    "budget_pairs": ((2, 24), (4, 18)),
}

PARTS = ("law", "depth", "budget")


# ---------------------------------------------------------------------------
# The exact reference: the radius-r estimator, in closed form
# ---------------------------------------------------------------------------

def radius_r_error(sigma0: np.ndarray, r: int, t: float) -> float:
    """RMS error of the optimal radius-``r`` linear denoiser, at interior sites.

    For a Gaussian prior the full posterior mean is ``alpha Sigma_0 Sigma_t^{-1} x``. The
    radius-``r`` estimator is the same object restricted to a window: for site ``i`` it uses
    only ``x_{i-r..i+r}``, so its coefficients are the corresponding row of the *window's*
    own ``Sigma_0 Sigma_t^{-1}``. Both are exact linear algebra -- no training, no grid, no
    Monte Carlo -- so this reference curve carries no estimation error of its own.

    Reported at interior sites only. Sites near the boundary have truncated windows and a
    genuinely easier problem (less to condition on, but also less signal), and mixing them in
    would blur the decay the experiment is trying to measure.
    """
    n = sigma0.shape[0]
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    st = sigma_t(sigma0, alpha, delta)

    full = alpha * sigma0 @ np.linalg.inv(st)          # exact posterior-mean operator
    interior = range(r, n - r) if r > 0 else range(n)

    errs = []
    for i in interior:
        lo, hi = max(0, i - r), min(n, i + r + 1)
        idx = np.arange(lo, hi)
        # Optimal linear estimator of a_i from x[idx]: Cov(a_i, x_idx) Cov(x_idx)^{-1}.
        cov_ax = alpha * sigma0[i, idx]
        cov_xx = st[np.ix_(idx, idx)]
        w = np.linalg.solve(cov_xx, cov_ax)

        # Error variance of the window estimator against the full posterior mean, computed
        # in closed form: Var(w'x - f'x) where f is the full operator's row i.
        f = full[i]
        g = np.zeros(n)
        g[idx] = w
        d = f - g
        errs.append(float(d @ st @ d))
    return float(np.sqrt(np.mean(errs)))


def fit_log_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope of ``log y`` on ``x``, with a standard error from the residuals."""
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    xx, yy = x[m], np.log(y[m])
    a = np.vstack([xx, np.ones_like(xx)]).T
    coef, *_ = np.linalg.lstsq(a, yy, rcond=None)
    resid = yy - a @ coef
    dof = max(len(xx) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(a.T @ a)
    return float(coef[0]), float(np.sqrt(cov[0, 0]))


def predicted_q(rho: float, t: float) -> float:
    """The closed-form decay base ``q`` of ledger item G12.

    ``J_d`` and ``beta`` are the diagonal and off-diagonal of the posterior precision of the
    Gaussian chain: ``J = (alpha^2/delta) I + Q_0`` with ``Q_0`` the tridiagonal clean
    precision.
    """
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    q0 = 1.0 - rho ** 2
    j_d = alpha ** 2 / delta + (1.0 + rho ** 2) / q0     # bulk diagonal
    beta = -rho / q0                                      # off-diagonal
    disc = j_d ** 2 - 4.0 * beta ** 2
    if disc <= 0:
        return float("nan")
    return float((j_d - np.sqrt(disc)) / (2.0 * abs(beta)))


# ---------------------------------------------------------------------------
# Chain-local network: depth is literally the information horizon
# ---------------------------------------------------------------------------

class ChainLocalNet:
    """Stack of width-3 shared convolutions over sites, with a per-site MLP channel mix.

    Each layer mixes a site with its two neighbours *only*. After ``d`` layers a site has
    seen exactly the window of radius ``d`` -- which is the entire point: the architecture's
    information horizon is its depth, by construction, so "depth" and "how far BP has
    propagated" are the same quantity.

    Zero padding at the boundary is used, and all reporting is restricted to interior sites
    (``depth <= i < n - depth``) so that padding never enters a reported number. A network
    reading padding would have an artificially easy boundary and could appear to beat its
    own horizon.
    """

    def __init__(self, depth: int, width: int, n_time: int, seed: int):
        rng = np.random.default_rng(seed)
        self.depth = depth
        self.width = width
        self.n_time = n_time
        self.W = []
        self.b = []
        # layer 0 consumes (3 sites + time features); later layers consume (3 * width)
        in_dim = 3 + n_time
        for d in range(depth):
            out_dim = width
            self.W.append(rng.normal(0, np.sqrt(2.0 / (in_dim + out_dim)), (in_dim, out_dim)))
            self.b.append(np.zeros(out_dim))
            in_dim = 3 * width
        self.W_out = rng.normal(0, np.sqrt(1.0 / width), (width, 1))
        self.b_out = np.zeros(1)

    @property
    def n_params(self) -> int:
        return (sum(w.size for w in self.W) + sum(v.size for v in self.b)
                + self.W_out.size + self.b_out.size)

    @staticmethod
    def _neighbourhoods(h: np.ndarray) -> np.ndarray:
        """(B, n, C) -> (B, n, 3C): each site concatenated with its two neighbours."""
        left = np.concatenate([np.zeros_like(h[:, :1]), h[:, :-1]], axis=1)
        right = np.concatenate([h[:, 1:], np.zeros_like(h[:, :1])], axis=1)
        return np.concatenate([left, h, right], axis=2)

    def forward(self, x: np.ndarray, tf: np.ndarray):
        """x: (B, n).  tf: (B, n_time).  Returns (B, n) and the activation cache."""
        b, n = x.shape
        h = self._neighbourhoods(x[:, :, None])                      # (B, n, 3)
        h = np.concatenate([h, np.repeat(tf[:, None, :], n, axis=1)], axis=2)
        cache = []
        for d in range(self.depth):
            z = h @ self.W[d] + self.b[d]
            a = np.tanh(z)
            cache.append((h, a))
            h = self._neighbourhoods(a) if d < self.depth - 1 else a
        out = (h @ self.W_out + self.b_out)[:, :, 0]
        return out, cache


def train_chain_local(
    A: np.ndarray, t_values, depth: int, width: int, cfg: dict, seed: int
) -> ChainLocalNet:
    """Denoising score matching, eps-parameterization, plain SGD with Adam.

    Gradients are obtained by finite-difference-free manual backprop through the same
    neighbourhood operator used in the forward pass; correctness is asserted by
    ``tests/test_depth_law.py`` against numerical differentiation, because a wrong gradient
    here would train to a worse optimum and be indistinguishable from "depth does not help".
    """
    rng = rng_for("exp17-train", seed, depth, width)
    net = ChainLocalNet(depth, width, n_time=4, seed=seed)

    mW = [np.zeros_like(w) for w in net.W]
    vW = [np.zeros_like(w) for w in net.W]
    mb = [np.zeros_like(v) for v in net.b]
    vb = [np.zeros_like(v) for v in net.b]
    mo, vo = np.zeros_like(net.W_out), np.zeros_like(net.W_out)
    mbo, vbo = np.zeros_like(net.b_out), np.zeros_like(net.b_out)
    b1, b2, eps = 0.9, 0.999, 1e-8

    n_chains, n = A.shape
    for step in range(1, cfg["n_steps"] + 1):
        idx = rng.integers(0, n_chains, cfg["batch_size"])
        a = A[idx]
        t = np.array([t_values[i] for i in rng.integers(0, len(t_values), len(idx))])
        alpha = np.exp(-t)[:, None]
        delta = (1.0 - np.exp(-2.0 * t))[:, None]
        z = rng.standard_normal(a.shape)
        x = alpha * a + np.sqrt(delta) * z
        tf = np.stack([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)], axis=1)

        pred, cache = net.forward(x, tf)
        diff = pred - z
        # Interior-only loss: the boundary sites see zero padding and are not part of the
        # claim being tested.
        m = np.zeros_like(diff)
        m[:, depth:n - depth] = 1.0
        denom = max(float(m.sum()), 1.0)
        g_out = 2.0 * diff * m / denom

        # --- backward
        h_last, a_last = cache[-1]
        gW_out = a_last.reshape(-1, net.width).T @ g_out.reshape(-1, 1)
        gb_out = g_out.sum().reshape(1)
        delta_h = g_out[:, :, None] @ net.W_out.T             # (B, n, width)

        gW = [None] * net.depth
        gb = [None] * net.depth
        for d in range(net.depth - 1, -1, -1):
            h_in, a_d = cache[d]
            dz = delta_h * (1.0 - a_d ** 2)
            gW[d] = h_in.reshape(-1, h_in.shape[2]).T @ dz.reshape(-1, net.width)
            gb[d] = dz.sum(axis=(0, 1))
            if d > 0:
                back = dz @ net.W[d].T                        # (B, n, 3*width_prev)
                w = net.width
                # scatter the neighbourhood gradient back onto sites
                acc = np.zeros((back.shape[0], back.shape[1], w))
                acc += back[:, :, w:2 * w]
                acc[:, :-1] += back[:, 1:, :w]
                acc[:, 1:] += back[:, :-1, 2 * w:]
                delta_h = acc

        # --- Adam
        for d in range(net.depth):
            for g, mm, vv, par in ((gW[d], mW, vW, net.W), (gb[d], mb, vb, net.b)):
                mm[d] = b1 * mm[d] + (1 - b1) * g
                vv[d] = b2 * vv[d] + (1 - b2) * g ** 2
                par[d] -= cfg["lr"] * (mm[d] / (1 - b1 ** step)) / (
                    np.sqrt(vv[d] / (1 - b2 ** step)) + eps)
        mo = b1 * mo + (1 - b1) * gW_out
        vo = b2 * vo + (1 - b2) * gW_out ** 2
        net.W_out -= cfg["lr"] * (mo / (1 - b1 ** step)) / (np.sqrt(vo / (1 - b2 ** step)) + eps)
        mbo = b1 * mbo + (1 - b1) * gb_out
        vbo = b2 * vbo + (1 - b2) * gb_out ** 2
        net.b_out -= cfg["lr"] * (mbo / (1 - b1 ** step)) / (np.sqrt(vbo / (1 - b2 ** step)) + eps)

    return net


def evaluate_net(net, A, sigma0, t, depth) -> float:
    """RMS deviation from the exact posterior mean, interior sites only."""
    rng = rng_for("exp17-eval", depth)
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    x = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    # exact_gaussian_posterior_mean is column-oriented: (n_sites, batch).
    m_star = exact_gaussian_posterior_mean(x.T, sigma0, alpha, delta).T
    tf = np.stack([np.sin([t]), np.cos([t]), np.sin([2 * t]), np.cos([2 * t])], axis=1)
    tf = np.repeat(tf, x.shape[0], axis=0)
    zhat, _ = net.forward(x, tf)
    m_hat = (x - np.sqrt(delta) * zhat) / alpha
    n = A.shape[1]
    sl = slice(depth, n - depth)
    return float(np.sqrt(np.mean((m_hat[:, sl] - m_star[:, sl]) ** 2)))


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def run_law(cfg, out_dir):
    """Verify the reference curve before anything is compared against it."""
    rows = []
    sigma0 = chain_covariance(cfg["n_sites"], cfg["rho"])
    for t in cfg["t_probe"]:
        radii = np.array([r for r in cfg["radii"] if r <= cfg["n_sites"] // 2 - 1])
        errs = np.array([radius_r_error(sigma0, int(r), t) for r in radii])
        slope, se = fit_log_slope(radii.astype(float), errs)
        q = predicted_q(cfg["rho"], t)
        pred_slope = np.log(q) if np.isfinite(q) and q > 0 else float("nan")
        print(f"  t={t:4.2f}  fitted log-slope {slope:+.4f}+-{se:.4f}   "
              f"predicted log(q)={pred_slope:+.4f}   q={q:.4f}", flush=True)
        for r, e in zip(radii, errs):
            rows.append({"t": t, "radius": int(r), "rms_error": e,
                         "fitted_slope": slope, "fitted_slope_se": se,
                         "predicted_log_q": pred_slope, "q": q})
    write_csv(out_dir / "law.csv", rows)
    return rows


def run_depth(cfg, out_dir):
    rows = []
    sigma0 = chain_covariance(cfg["n_sites"], cfg["rho"])
    rng = rng_for("exp17-data", cfg["seed_offset"])
    L = np.linalg.cholesky(sigma0)
    A_tr = rng.standard_normal((cfg["n_train"], cfg["n_sites"])) @ L.T
    A_te = rng.standard_normal((cfg["n_test"], cfg["n_sites"])) @ L.T

    for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
        for d in cfg["depths"]:
            net = train_chain_local(A_tr, cfg["t_probe"], d, cfg["width"], cfg, seed)
            for t in cfg["t_probe"]:
                got = evaluate_net(net, A_te, sigma0, t, d)
                ref = radius_r_error(sigma0, d, t)
                rows.append({"seed": seed, "depth": d, "width": cfg["width"],
                             "n_params": net.n_params, "t": t,
                             "net_rms": got, "radius_d_rms": ref,
                             "ratio_net_over_radius": got / ref if ref > 0 else np.nan})
                print(f"  seed={seed} d={d:2d} t={t:4.2f}  net={got:.5f}  "
                      f"radius-{d}={ref:.5f}  ratio={got / ref if ref > 0 else float('nan'):6.2f}",
                      flush=True)
    write_csv(out_dir / "depth.csv", rows)
    return rows


def run_budget(cfg, out_dir):
    """Depth vs width at (roughly) fixed parameter count."""
    rows = []
    sigma0 = chain_covariance(cfg["n_sites"], cfg["rho"])
    rng = rng_for("exp17-data", cfg["seed_offset"])
    L = np.linalg.cholesky(sigma0)
    A_tr = rng.standard_normal((cfg["n_train"], cfg["n_sites"])) @ L.T
    A_te = rng.standard_normal((cfg["n_test"], cfg["n_sites"])) @ L.T

    for seed in range(cfg["seed_offset"], cfg["seed_offset"] + cfg["n_seed"]):
        for d, w in cfg["budget_pairs"]:
            net = train_chain_local(A_tr, cfg["t_probe"], d, w, cfg, seed)
            for t in cfg["t_probe"]:
                got = evaluate_net(net, A_te, sigma0, t, d)
                ref = radius_r_error(sigma0, d, t)
                rows.append({"seed": seed, "depth": d, "width": w,
                             "n_params": net.n_params, "t": t,
                             "net_rms": got, "radius_d_rms": ref})
                print(f"  seed={seed} d={d:2d} w={w:3d} params={net.n_params:6d} "
                      f"t={t:4.2f} net={got:.5f} radius-{d}={ref:.5f}", flush=True)
    write_csv(out_dir / "budget.csv", rows)
    return rows


def main() -> None:
    parser = experiment_parser("exp_17_depth_law", __doc__)
    args = parser.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return

    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = PARTS if args.only is None else tuple(p.strip() for p in args.only.split(","))
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown part(s): {sorted(unknown)}; choose from {PARTS}")

    out_dir = ensure_dir(args.output_dir)
    for part in parts:
        print(f"\n=== part: {part} ===", flush=True)
        {"law": run_law, "depth": run_depth, "budget": run_budget}[part](cfg, out_dir)

    write_json(out_dir / f"params_{'_'.join(parts)}.json",
               {"settings": cfg, "parts": list(parts), **provenance()})


if __name__ == "__main__":
    main()
