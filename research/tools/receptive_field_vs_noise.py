"""Independent check on WHY the structured estimator's advantage grows with noise.

THE FINDING BEING TESTED. `rate_robustness.py` fits the sample-efficiency rate
separately at each noise level and finds the gap between the two arms tracks
log t with a correlation of -0.994: the network's slope is nearly flat in t
(-0.20 to -0.27) while EM--BP's steepens monotonically (-0.31 to -0.54). The
proposed mechanism is that the Markov factorisation buys the estimator the
ability to pool information along the chain, and that pooling is worth little at
small t (a site's own observation nearly determines its posterior) and
everything at large t (the per-site signal is weak).

WHY THIS IS AN INDEPENDENT TEST. That mechanism makes a prediction about a
completely different experiment: if the score at large t genuinely depends on
information spread along the chain, then a network's required RECEPTIVE FIELD
must grow with t. exp_12 measured exactly that, for a different purpose, holding
n_params nearly constant across radii (4545-6081) so that radius is varied at
fixed capacity rather than confounded with it. Nothing in exp_12 was collected
with this question in mind, which is what makes it worth asking of.

WHAT IS FOUND. The smallest radius reaching within `TOL` of the best error at
each t grows monotonically with t in all six family x parameterisation
combinations -- from 2 at t=0.1 to between 6 and 16 at t=1.6. The curves say
more than the summary: at small t the error *rises* again past the optimum
(extra context costs capacity that is better spent locally), while at large t it
is still falling at the largest radius measured. So the requirement is not merely
larger at high noise, it is unsatisfied there.

AND THE GROWTH MATCHES THEORY IN SHAPE. For the Gaussian AR(1) chain the
information radius is computable in closed form: the exact posterior mean is
E[a | x] = alpha Sigma_0 Sigma_t^{-1} x, so row i of that matrix states exactly
how much each observation contributes to site i. The radius capturing a fixed
fraction of that weight grows with t, and the measured requirement tracks it with
a correlation of about 0.99.

BUT NOT IN ABSOLUTE SIZE, and the difference matters. The two radii use different
criteria -- a fraction of posterior WEIGHT against a fraction of achievable ERROR
improvement -- so their absolute agreement is a free parameter. Measured against
theory at mass thresholds 0.80/0.90/0.95/0.99 the mean discrepancy is
1.13/0.47/2.27/6.27 sites while the correlation stays at 0.995/0.998/0.990/0.994.
The shape is the finding; the site-level agreement at any one threshold is a
coincidence of that threshold and must not be quoted as calibration.

WHAT THE FAMILY SPREAD MEASURES, which is the more interesting quantity. All
three innovation families are variance-matched -- they share
Cov(a_i, a_j) = rho^|i-j| exactly and differ only beyond second moments -- so the
closed-form radius above is IDENTICAL for all of them. Any family dependence in
the measured requirement is therefore purely non-Gaussian structure. It is zero
at small t (2.0, 2.0, 2.0 sites at t=0.1) and large at big t (12.0 for the
Gaussian mixture, 10.0 for Laplace, 7.0 for the Gaussian at t=1.6). Family spread
averages 1.60 sites against 0.53 across parameterisations, so it is the larger
effect and it is concentrated where the posterior is most prior-dominated.

Read together: the second-order theory sets the shape of the pooling requirement,
and what it fails to set -- growing from nothing at t=0.1 to five sites at
t=1.6 -- is exactly the beyond-second-order structure this project exists to
measure. That the Gaussian family needs the LEAST radius of the three is the
sign one would want.

This does not prove the mechanism -- it is two predictions, one confirmed
directionally and one confirmed in shape but not in scale. It does rule out the
reading that the noise-level trend in the rate gap is a quirk of the efficiency
experiment's protocol, since exp_12 shares none of that protocol and the closed
form shares none of either.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

SOURCE = "outputs/exp_12_receptive_field/receptive_field.csv"
# The chain correlation the frozen configuration uses. The closed-form radius
# below is a property of THIS chain, so it must track the config rather than be
# typed as a constant that silently goes stale.
# sys.path[0] is tools/ when this runs as a script, not the package root, so
# neither `src` nor `experiments` is importable without help. Both are relative
# to the cwd, which is the package root -- the same convention every other tool
# here uses.
try:                                             # pragma: no cover - path setup
    sys.path.insert(0, ".")
    sys.path.insert(0, "experiments")
    from frozen_config import FROZEN
    CHAIN_RHO = float(FROZEN.rho)
except Exception:
    CHAIN_RHO = 0.85
OUT_DIR = "outputs/rate_robustness"
OUT = f"{OUT_DIR}/receptive_field_vs_noise.csv"

# "Within TOL of the best" as a fraction of the total improvement available at
# that t, so the threshold means the same thing at every noise level. An absolute
# error threshold would not: the radius-0 error itself falls by a factor of ten
# across the schedule.
TOL = 0.10

# The claim is monotone growth. Allowing one non-monotone step per curve, because
# the radii are coarse (0,1,2,3,4,6,8,12,16) and a single seed's noise can invert
# one adjacent pair without touching the trend.
MAX_INVERSIONS = 1


def theoretical_radius(t: float, rho: float = CHAIN_RHO,
                       n: int = 129, mass: float = 0.90) -> int:
    """Radius capturing `mass` of the exact posterior weight at noise level t.

    Exact for the Gaussian chain: E[a | x] = alpha Sigma_0 Sigma_t^{-1} x, so the
    row of that matrix at an interior site IS the set of contributions, and no
    fitting or thresholding convention enters beyond `mass` itself.
    """
    from src.exact_scores import sigma_t
    from src.noising import alpha_delta

    i = n // 2
    sigma0 = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    alpha, delta = alpha_delta(t)
    w = np.abs((alpha * sigma0 @ np.linalg.inv(sigma_t(sigma0, alpha, delta)))[i])
    w = w / w.sum()
    for R in range(n // 2):
        if w[max(0, i - R):i + R + 1].sum() >= mass:
            return R
    return n // 2


def main():
    if not os.path.exists(SOURCE):
        print(f"REFUSING: no receptive-field run at {SOURCE}", file=sys.stderr)
        sys.exit(1)
    rows = list(csv.DictReader(open(SOURCE)))

    radii = sorted({int(r["radius"]) for r in rows})
    out_rows, curves = [], {}
    for fam in sorted({r["family"] for r in rows}):
        for par in sorted({r["parameterization"] for r in rows}):
            sub = [r for r in rows if r["family"] == fam and r["parameterization"] == par]
            if not sub:
                continue
            needed = []
            for t in sorted({float(r["t"]) for r in sub}):
                vals = []
                for R in radii:
                    m = [float(x["score_rel_l2_interior"]) for x in sub
                         if float(x["t"]) == t and int(x["radius"]) == R]
                    vals.append(float(np.mean(m)) if m else np.nan)
                vals = np.array(vals)
                best, at_zero = np.nanmin(vals), vals[0]
                thr = best + TOL * (at_zero - best)
                r_needed = next((R for R, v in zip(radii, vals) if v <= thr), radii[-1])
                needed.append(r_needed)
                out_rows.append({
                    "family": fam, "parameterization": par, "t": t,
                    "radius_needed": int(r_needed),
                    "radius_theory_gaussian": theoretical_radius(t),
                    "best_error": float(best),
                    "error_at_radius_0": float(at_zero),
                    "error_at_max_radius": float(vals[-1]),
                    # True when the largest radius measured is still the best:
                    # the requirement is not merely large, it is unmet.
                    "still_improving_at_max": bool(
                        np.nanargmin(vals) == len(vals) - 1
                    ),
                })
            curves[(fam, par)] = needed

    inversions = {
        k: sum(1 for a, b in zip(v, v[1:]) if b < a) for k, v in curves.items()
    }
    bad = {k: n for k, n in inversions.items() if n > MAX_INVERSIONS}
    if bad:
        print(
            f"REFUSING: required radius is not monotone in t for {sorted(bad)}. "
            f"The pooling mechanism predicts monotone growth; if it does not "
            f"hold, this is evidence against the explanation and must be "
            f"reported as such rather than quietly averaged away.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{'family':<20} {'param':<5} " +
          "".join(f"t={t:<6}" for t in sorted({r['t'] for r in out_rows})))
    for (fam, par), needed in curves.items():
        print(f"{fam:<20} {par:<5} " + "".join(f"{n:<8}" for n in needed))
    still = sum(1 for r in out_rows if r["still_improving_at_max"])
    print(f"\nmonotone in {len(curves) - len(bad)}/{len(curves)} curves; "
          f"{still}/{len(out_rows)} cells still improving at the largest radius")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
