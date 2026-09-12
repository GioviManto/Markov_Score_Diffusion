"""Experiment 28 -- Rung 4. Can EM recover dynamics that the marginals cannot see?

The question
------------
Rungs 0-3 all live on the AR(1) chain, where the thing being estimated (rho, an
innovation density) is visible in the data's low-order statistics. This rung
asks the same estimation question in a model where it is not.

The rotating ring (`src/ring.py`, Problem 1 of `research/board-3problems`): a
point starts on a ring of radius ~1 at a uniformly random angle and takes a
rotating random walk, `z_{u+1} = R_psi z_u + sigma eta_u`. The rotation angle
psi is drawn per trajectory from an unknown p(psi), and we observe whole
trajectories through the OU channel.

Two structural facts set the experiment up:

  * **The rotation is a gauge.** `P_t(x | psi) = P_t^0(U_psi x)` -- rotate the
    data back and evaluate the psi = 0 density. So the psi-conditional
    likelihood is closed form at any T, with no network and no Monte Carlo.

  * **Marginal blindness (a theorem).** `z_u =d z_0 + sigma sqrt(u) xi` for
    every psi, so every single-frame marginal is psi-free at every noise level.
    A per-frame model carries EXACTLY ZERO information about the rotation.

What is run
-----------
EM over p(psi) on a grid of angles, in two arms that differ ONLY in which
likelihood scores a trajectory:

    joint     r_mu(psi) ∝ p(psi) * P_t(x_mu | psi)      -- the closed form above
    marginal  r_mu(psi) ∝ p(psi) * prod_u P_t(x_mu,u)   -- the blind model

The marginal arm is not a strawman baseline. It is the theorem's own control:
its likelihood is provably constant in psi, so its responsibilities equal
p(psi) exactly, the M-step is the identity, and p(psi) can never leave its
initialisation. If the run shows anything else, the implementation is wrong.

One structural economy worth noting, because it mirrors the chain model. The
psi-conditional log-likelihood does not depend on p(psi) -- the parameter being
estimated. So the (M x n_psi) matrix of log-densities is computed ONCE and
every EM iteration is a reweighting of it. This is the same sufficiency that
makes `Xi` a one-shot statistic at Rungs 1-3: the E-step touches the data once,
and the M-step never revisits it.

    python3 experiments/exp_28_ring_em.py --list-parts
    python3 experiments/exp_28_ring_em.py --only recovery --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import apply_overrides, experiment_parser, provenance, select_parts
from frozen_config import FROZEN, FROZEN_ROOT
from src.ring import (
    RingConfig,
    log_pt_conditional,
    log_pt_marginal,
    noise,
    sample_trajectories,
)
from src.utils import ensure_dir, write_csv, write_json

# Two species: half the trajectories rotate by +omega, half by -omega. This is
# the sharpest form of the question -- the two species have identical marginals
# at every frame and every noise level, so nothing short of the joint can tell
# them apart, and a recovered p(psi) either has two spikes in the right places
# or it does not.
OMEGA = FROZEN.ring_psi_true          # pi/6 = 30 degrees


def psi_grid(n_psi: int) -> np.ndarray:
    """Angles on [-pi, pi), the period of the rotation."""
    return np.linspace(-np.pi, np.pi, n_psi, endpoint=False)


def true_pmf(grid: np.ndarray, omega: float) -> np.ndarray:
    """The two-species truth, binned onto the grid.

    Point masses land on the nearest grid point. The grid is fine enough
    (128 points over 2pi = 2.8 degrees) that this is a faithful rendering, and
    the same binning is applied to the truth and to the estimate, so the
    comparison is like with like.
    """
    pmf = np.zeros(grid.size)
    for psi, mass in ((+omega, 0.5), (-omega, 0.5)):
        pmf[int(np.argmin(np.abs(grid - psi)))] += mass
    return pmf


def conditional_loglik(X: np.ndarray, t: float, grid: np.ndarray,
                       cfg: RingConfig, arm: str) -> np.ndarray:
    """The (M x n_psi) matrix of log P(x_mu | psi).

    Computed once per dataset: it does not depend on p(psi).

    For `arm="marginal"` every column is the same psi-free number, which is the
    point -- we build the full matrix anyway rather than special-casing it, so
    that the two arms run through identical EM code and any difference in the
    result is a difference in the likelihood and not in the optimiser.
    """
    if arm == "joint":
        return np.stack([log_pt_conditional(X, t, psi, cfg) for psi in grid], axis=1)
    if arm == "marginal":
        blind = log_pt_marginal(X, t, cfg)
        return np.repeat(blind[:, None], grid.size, axis=1)
    raise ValueError(f"unknown arm {arm!r}")


def run_em(log_lik: np.ndarray, n_iters: int, tol: float = 1e-12):
    """EM for a mixture over the rotation angle.

    E-step  r[mu, k] ∝ p[k] * exp(log_lik[mu, k])
    M-step  p[k]     <- mean_mu r[mu, k]

    Initialised uniform, which is the honest choice: it encodes no knowledge of
    where the species are, and for the marginal arm it is also the fixed point,
    so that arm visibly stands still.

    Returns (p, history) with history carrying the marginal log-likelihood per
    iteration -- monotone ascent is a property of EM and is asserted by the
    caller, not assumed.
    """
    n_obs, n_psi = log_lik.shape
    p = np.full(n_psi, 1.0 / n_psi)
    history = []

    for it in range(n_iters):
        z = log_lik + np.log(np.maximum(p, 1e-300))[None, :]
        shift = z.max(axis=1, keepdims=True)
        w = np.exp(z - shift)
        denom = w.sum(axis=1, keepdims=True)
        # Marginal log-likelihood, log sum_k p_k P(x|psi_k), summed over data.
        loglik = float(np.sum(np.log(denom[:, 0]) + shift[:, 0]))
        history.append(loglik)

        p_new = (w / denom).mean(axis=0)
        move = float(np.abs(p_new - p).max())
        p = p_new
        if move < tol:
            break

    return p, np.array(history)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def circular_spread(grid: np.ndarray, p: np.ndarray) -> float:
    """1 - R, the circular dispersion of the recovered law.

    For a two-spike law at +-omega this is a clean summary: it is 1 - cos(omega)
    when the mass sits where it should, and near 1 for a uniform p(psi). Used
    to say, in one number, whether the arm found structure at all.
    """
    return 1.0 - float(np.abs(np.sum(p * np.exp(1j * grid))))


def omega_estimate(grid: np.ndarray, p: np.ndarray) -> float:
    """Recover the rotation magnitude omega from p(psi), as a scalar.

    The truth is symmetric, p* = (delta_{+omega} + delta_{-omega})/2, so
    E|psi| = omega exactly. This is the parameter-recovery statement for this
    rung, and unlike a total-variation distance to two point masses it is not
    limited by the grid resolution: it interpolates between grid points, so it
    can improve with M after TV has hit its binning floor.

    A uniform p(psi) returns pi/2 (90 degrees), which is the null value.
    """
    return float(np.sum(p * np.abs(grid)))


# ---------------------------------------------------------------------------
# Rung 4a -- learn the confining potential
# ---------------------------------------------------------------------------
#
# The board writes the radial confinement as a quadratic well,
# V(r) = (r - r_star)^2 / 2 lam, so "learn the potential" means recover the pair
# (lam, r_star). Both are estimated by ascending the exact marginal
# log-likelihood from a random initialisation -- literally the procedure in
# Jerome's description of the task -- with a numerical gradient, which is honest
# for two parameters and avoids a hand-derived derivative that could be wrong in
# the same silent way the normaliser was.
#
# Unlike the rotation, this parameter IS visible to the marginals: frame u's
# radial law is the well convolved with a Gaussian of variance sigma^2 u. So the
# blind arm is expected to succeed here, and that is the point of running it.


def _potential_loglik(X, t, arm, lam, r_star, sigma, psi):
    cfg = RingConfig(lam=max(lam, 1e-4), sigma=sigma, r_star=r_star)
    if arm == "joint":
        return float(log_pt_conditional(X, t, psi, cfg).sum())
    return float(log_pt_marginal(X, t, cfg).sum())


def fit_potential(X, t, arm, sigma, psi, lam_init, r_init,
                  n_steps=300, step0=0.05, plateau_tol=1e-9, plateau_window=15):
    """Gradient ascent on (log lam, r_star), from a random initialisation.

    `log lam` rather than `lam` keeps the width positive without a constraint.
    Central differences for the gradient; backtracking on the step so a bad
    step cannot leave the likelihood lower than it started.

    Stops on a plateau of the objective -- total improvement over the last
    `plateau_window` steps below `plateau_tol` -- with `n_steps` as a cap rather
    than a target. The cap is deliberately past the plateau: at 25 steps this
    fit reports lam = 0.075 against a truth of 0.05 and looks perfectly
    well-behaved while doing so.

    `plateau_tol` is PER SEQUENCE. The objective is a sum over the rows of `X`,
    so an absolute threshold is a different convergence criterion at every
    sample size -- eight times stricter at n=4096 than at n=512, purely because
    the sum has more terms. That made larger n run longer for no statistical
    reason, which is precisely the axis Rung 4a reports its scaling along, so
    the stopping rule was confounded with the measurement. Scaling by len(X)
    asks the same question of every size: has the fit stopped improving by more
    than `plateau_tol` nats per sequence over the last `plateau_window` steps?

    Returns (lam_hat, r_star_hat, history).
    """
    theta = np.array([np.log(lam_init), r_init])
    f = lambda th: _potential_loglik(X, t, arm, np.exp(th[0]), th[1], sigma, psi)
    cur = f(theta)
    hist = [cur]
    step = step0
    tol = plateau_tol * len(X)

    for _ in range(n_steps):
        if len(hist) > plateau_window and hist[-1] - hist[-1 - plateau_window] < tol:
            break
        grad = np.zeros(2)
        for k in range(2):
            h = 1e-4 * max(abs(theta[k]), 1.0)
            e = np.zeros(2); e[k] = h
            grad[k] = (f(theta + e) - f(theta - e)) / (2 * h)
        norm = np.linalg.norm(grad)
        if not np.isfinite(norm) or norm < 1e-12:
            break
        direction = grad / norm

        # Backtrack until the step actually increases the objective.
        accepted = False
        for _ in range(8):
            cand = theta + step * direction
            cand[1] = np.clip(cand[1], 0.05, 5.0)
            val = f(cand)
            if val > cur:
                theta, cur, accepted = cand, val, True
                step *= 1.3
                break
            step *= 0.5
        hist.append(cur)
        if not accepted:
            break

    return float(np.exp(theta[0])), float(theta[1]), np.array(hist)


def part_potential(cfg: dict, out: Path) -> list[dict]:
    """Rung 4a. Recover (lam, r_star) on both arms."""
    rows: list[dict] = []
    lam_true, r_true = cfg["ring_lambda"], cfg["ring_r_star"]
    ring_true = RingConfig(lam=lam_true, sigma=cfg["ring_sigma"], r_star=r_true)

    for seed in cfg["seeds"]:
        for n_obs in cfg["potential_sizes"]:
            rng = np.random.default_rng(seed)
            # ONE species, as in part_rotation. This was two (+-OMEGA shuffled)
            # until 2026-08-17, while fit_potential scores every trajectory at
            # the single scalar `psi` it is handed -- so half the data was
            # evaluated under the wrong rotation.
            #
            # The failure was not subtle once looked at, but it was invisible in
            # aggregate: the joint arm answered the misspecification by driving
            # r_star to the lower clip (0.05 in 52% of cells across eight seeds),
            # because a ring of radius zero is invariant under rotation and a
            # collapsed ring therefore pays no penalty for the wrong psi. The
            # blind arm, which never conditions on psi, was unaffected -- so the
            # sweep reported the marginal arm BEATING the joint arm, which is the
            # reverse of what Rung 4a exists to show.
            #
            # Matched species, measured at n=512, t=0.4665, three seeds:
            #   two species  joint  lam err 3.01-3.37   r* err 0.950 (pinned)
            #   one species  joint  lam err 0.07-0.24   r* err 0.003-0.042
            psis = np.full(n_obs, OMEGA)
            a, _ = sample_trajectories(n_obs, cfg["ring_n_frames"], psis, ring_true, rng)

            # Initialisation is random and away from the truth, so the well is
            # genuinely recovered rather than supplied.
            lam0 = float(np.exp(rng.uniform(np.log(0.01), np.log(0.30))))
            r0 = float(rng.uniform(0.6, 1.5))

            for t in cfg["potential_t"]:
                X = noise(a, t, rng)
                for arm in ("joint", "marginal"):
                    lam_hat, r_hat, hist = fit_potential(
                        X, t, arm, cfg["ring_sigma"], OMEGA, lam0, r0,
                        n_steps=cfg["potential_steps"],
                    )
                    rows.append({
                        "seed": seed, "n_obs": n_obs, "t": t, "arm": arm,
                        "lam_init": lam0, "r_star_init": r0,
                        "lam_hat": lam_hat, "lam_true": lam_true,
                        "lam_rel_err": abs(lam_hat - lam_true) / lam_true,
                        "r_star_hat": r_hat, "r_star_true": r_true,
                        "r_star_abs_err": abs(r_hat - r_true),
                        "n_steps": int(hist.size),
                        # Hit the cap rather than the plateau => not converged,
                        # and the estimate must not be read as one.
                        "hit_cap": bool(hist.size >= cfg["potential_steps"]),
                        "loglik_init": float(hist[0]),
                        "loglik_final": float(hist[-1]),
                        "monotone": bool(np.all(np.diff(hist) >= -1e-9)),
                    })
    return rows


# ---------------------------------------------------------------------------
# Rung 4b -- learn a single rotation
# ---------------------------------------------------------------------------
#
# One scalar, fewer parameters than the potential above -- and by
# Theorem 1 no single-frame marginal carries any information about it. This is
# the pair that makes the point: same model, same estimator, one parameter each,
# and the blind arm succeeds on one and cannot move on the other.


def fit_rotation(X, t, arm, cfg_ring, n_grid=181):
    """Profile the likelihood over a single psi, then refine parabolically.

    Returns (psi_hat, is_flat, profile_range). `is_flat` is the diagnostic that
    matters: on the marginal arm the profile is constant in psi to machine
    precision, so there is nothing to maximise and no estimate to report.
    """
    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    if arm == "joint":
        prof = np.array([log_pt_conditional(X, t, p, cfg_ring).sum() for p in grid])
    else:
        blind = float(log_pt_marginal(X, t, cfg_ring).sum())
        prof = np.full(grid.size, blind)

    spread = float(prof.max() - prof.min())
    # Scale-free flatness test: a profile whose whole range is at the level of
    # floating-point noise in its own magnitude carries no information.
    if spread <= 1e-9 * max(abs(float(prof.mean())), 1.0):
        return float("nan"), True, spread

    k = int(np.argmax(prof))
    # Parabolic refinement through the three points around the peak, so the
    # estimate is not limited by the grid.
    km, kp = (k - 1) % grid.size, (k + 1) % grid.size
    y0, y1, y2 = prof[km], prof[k], prof[kp]
    denom = y0 - 2 * y1 + y2
    dx = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    step = grid[1] - grid[0]
    return float(grid[k] + np.clip(dx, -1, 1) * step), False, spread


def part_rotation(cfg: dict, out: Path) -> list[dict]:
    """Rung 4b. Recover a single psi on both arms.

    All trajectories share one rotation here -- that is what makes it the
    one-parameter case, as opposed to the two-species mixture of `part_recovery`.
    """
    rows: list[dict] = []
    ring = RingConfig(lam=cfg["ring_lambda"], sigma=cfg["ring_sigma"],
                      r_star=cfg["ring_r_star"])

    for seed in cfg["seeds"]:
        for n_obs in cfg["potential_sizes"]:
            rng = np.random.default_rng(seed)
            a, _ = sample_trajectories(n_obs, cfg["ring_n_frames"],
                                       np.full(n_obs, OMEGA), ring, rng)
            for t in cfg["potential_t"]:
                X = noise(a, t, rng)
                for arm in ("joint", "marginal"):
                    psi_hat, flat, spread = fit_rotation(X, t, arm, ring)
                    err = float("nan") if flat else abs(psi_hat - OMEGA)
                    rows.append({
                        "seed": seed, "n_obs": n_obs, "t": t, "arm": arm,
                        "psi_hat_deg": "" if flat else np.degrees(psi_hat),
                        "psi_true_deg": np.degrees(OMEGA),
                        "psi_err_deg": "" if flat else np.degrees(err),
                        "is_flat": flat,
                        "profile_range_nats": spread,
                    })
    return rows


def part_recovery(cfg: dict, out: Path) -> list[dict]:
    """Recover p(psi) by EM, joint arm against the blind marginal arm."""
    ring = RingConfig(lam=cfg["ring_lambda"], sigma=cfg["ring_sigma"])
    grid = psi_grid(cfg["ring_n_psi"])
    truth = true_pmf(grid, OMEGA)
    rows: list[dict] = []

    for seed in cfg["seeds"]:
        for n_obs in cfg["sizes"]:
            rng = np.random.default_rng(seed)
            # Half the trajectories in each species, shuffled so that species
            # membership is not confounded with position in the array.
            psis = np.where(np.arange(n_obs) < n_obs // 2, +OMEGA, -OMEGA)
            rng.shuffle(psis)
            a, _ = sample_trajectories(n_obs, cfg["ring_n_frames"], psis, ring, rng)

            for t in cfg["t_grid"]:
                X = noise(a, t, rng)
                for arm in ("joint", "marginal"):
                    log_lik = conditional_loglik(X, t, grid, ring, arm)
                    p, hist = run_em(log_lik, cfg["em_iters"])

                    ascent = float(np.min(np.diff(hist))) if hist.size > 1 else 0.0
                    moved = total_variation(p, np.full(grid.size, 1.0 / grid.size))
                    # A flat p(psi) has no mode. Reporting argmax of a constant
                    # array would print an arbitrary grid point that looks like
                    # an estimate, so it is recorded as missing instead.
                    flat = moved < 1e-12
                    omega_hat = omega_estimate(grid, p)

                    rows.append({
                        "seed": seed,
                        "n_obs": n_obs,
                        "t": t,
                        "arm": arm,
                        # Primary: a scalar parameter, not grid-limited.
                        "omega_hat_deg": np.degrees(omega_hat),
                        "omega_err_deg": np.degrees(abs(omega_hat - OMEGA)),
                        # Secondary: shape of the recovered law. TV against two
                        # point masses is floored by the grid binning, so it is
                        # reported but is not the headline.
                        "tv_to_truth": total_variation(p, truth),
                        "mass_near_pm_omega": float(
                            p[np.abs(np.abs(grid) - OMEGA) < 0.10].sum()
                        ),
                        "spread": circular_spread(grid, p),
                        # How far the estimate moved from its uniform start. For
                        # the marginal arm the theorem forces this to be zero.
                        "moved_from_uniform": moved,
                        "is_flat": flat,
                        "mode_psi_deg": "" if flat else np.degrees(float(grid[int(np.argmax(p))])),
                        "n_iters": int(hist.size),
                        "min_loglik_increase": ascent,
                        "final_loglik": float(hist[-1]),
                    })
    return rows


def part_density(cfg: dict, out: Path) -> list[dict]:
    """The recovered p(psi) itself, on the grid -- this is the figure.

    One representative cell (largest M, smallest t, first seed) for each arm,
    so the plot shows what EM actually returns rather than a summary of it.
    """
    ring = RingConfig(lam=cfg["ring_lambda"], sigma=cfg["ring_sigma"])
    grid = psi_grid(cfg["ring_n_psi"])
    truth = true_pmf(grid, OMEGA)
    seed = cfg["seeds"][0]
    n_obs = max(cfg["sizes"])
    rows: list[dict] = []

    for t in cfg["density_t"]:
        rng = np.random.default_rng(seed)
        psis = np.where(np.arange(n_obs) < n_obs // 2, +OMEGA, -OMEGA)
        rng.shuffle(psis)
        a, _ = sample_trajectories(n_obs, cfg["ring_n_frames"], psis, ring, rng)
        X = noise(a, t, rng)

        for arm in ("joint", "marginal"):
            p, _ = run_em(conditional_loglik(X, t, grid, ring, arm),
                          cfg["em_iters"])
            for k, psi in enumerate(grid):
                rows.append({
                    "t": t, "arm": arm, "n_obs": n_obs, "seed": seed,
                    "psi": float(psi), "p_hat": float(p[k]),
                    "p_true": float(truth[k]),
                })
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_28_ring_em",
        "Rung 4: EM recovery of rotation dynamics that per-frame marginals cannot see.",
    )
    args = parser.parse_args()

    settings = {
        "seeds": list(FROZEN.seeds()),
        "sizes": list(FROZEN.sizes),
        "t_grid": list(FROZEN.t_grid),
        "density_t": [0.05, 0.4665, 1.425],
        # Rungs 4a/4b sweep a decay curve rather than a fine schedule, so they
        # use every other level of the frozen twelve and five of the seven
        # budgets. Stated here rather than silently: the reduction is to keep
        # a 300-step gradient fit per cell inside a cluster walltime, and the
        # remaining grid still spans the full range on both axes.
        "potential_t": list(FROZEN.t_grid[::2]),
        "potential_sizes": [128, 512, 1024, 2048, 4096],
        "ring_n_frames": FROZEN.ring_n_frames,
        "ring_sigma": FROZEN.ring_sigma,
        "ring_lambda": FROZEN.ring_lambda,
        "ring_r_star": FROZEN.ring_r_star,
        "ring_n_psi": FROZEN.ring_n_psi,
        "em_iters": 200,  # frozen-exempt: EM over p(psi), not the innovation kernel -- a 128-cell discrete latent, converged by ~40 (exp_28 traces)
        # Measured: at 25 steps the estimate is still at lam = 0.075 against a
        # truth of 0.05; the likelihood plateaus by ~150 and the estimate is
        # settled by ~300 (lam = 0.0493, r_star = 0.9966 at n = 512). Stopping
        # early here reproduces, in miniature, exactly the under-iteration
        # mistake that this whole rebuild exists to correct -- so the budget is
        # set past the plateau rather than at it.
        "potential_steps": 300,
    }
    if args.quick:
        settings.update(
            seeds=list(FROZEN.seeds())[:2],
            sizes=[128, 512],
            t_grid=[0.05, 0.4665],
            density_t=[0.05],
            ring_n_psi=64,
            potential_t=[0.05, 0.4665],
            potential_sizes=[128, 512],
            em_iters=60,
            potential_steps=40,      # smoke only -- NOT converged, see above
        )
    cfg = apply_overrides(settings, args.set)

    # Rung 4 is one model with three targets, ordered by what the marginals can
    # see. Each part runs BOTH arms, so the three CSVs together are a 3x2 design.
    parts = {
        "potential": ("4a: learn the quadratic well (lam, r_star) -- marginals CAN see it",
                      lambda out: write_csv(out / "ring_potential.csv",
                                            part_potential(cfg, out))),
        "rotation": ("4b: learn one rotation psi -- marginals CANNOT see it",
                     lambda out: write_csv(out / "ring_rotation.csv",
                                           part_rotation(cfg, out))),
        "recovery": ("4c: EM recovery of p(psi), joint vs blind marginal arm",
                     lambda out: write_csv(out / "ring_recovery.csv",
                                           part_recovery(cfg, out))),
        "density": ("the recovered p(psi) on the angular grid, for plotting",
                    lambda out: write_csv(out / "ring_density.csv",
                                          part_density(cfg, out))),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir if args.output_dir != parser.get_default("output_dir")
                     else FROZEN_ROOT / "exp_28_ring_em")

    tag = "_".join(selected) if args.only else "all"
    write_json(out / f"params_{tag}.json", {
        "omega_true_rad": OMEGA,
        "omega_true_deg": np.degrees(OMEGA),
        "quick": args.quick,
        "parts": list(selected),
        "overrides": args.set,
        **cfg,
        **provenance(),
    })

    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
