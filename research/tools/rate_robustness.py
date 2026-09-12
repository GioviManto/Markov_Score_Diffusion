"""Try to break the convergence-rate finding, and measure what a budget is worth.

BACKGROUND. `make_rate_analysis.py` reports that EM--BP's relative denoising error
falls faster with nseq than the network's: a paired log--log slope difference of
0.119 +/- 0.011, steeper for EM--BP in all 16 seeds. That number rests on ONE
error functional (relative score error) measured under ONE protocol (budget chosen
on a validation bundle). A finding that exists only under one pair of choices is
not a finding, so this reads the same cells four other ways.

    selected   score error, budget chosen on validation   -- the original
    oracle     score error, budget chosen on TEST         -- removes selection
                                                             noise entirely
    at_cap     score error at the MAXIMUM budget          -- removes selection
                                                             ITSELF: no budget is
                                                             chosen for either arm
    mean       posterior-mean error, validation-selected  -- a different error
                                                             functional, same fits

`at_cap` is the load-bearing one. Every objection to the rate finding so far has
been an objection about how a budget got chosen -- diverging cap-hit rates, one
arm more censored than the other, validation selection favouring an arm. At the
cap nothing is chosen for anybody, so none of those can operate.

WHAT A BUDGET IS WORTH, which falls out of the same comparison. The ratio
(error at cap) / (error at the selected budget) is the price of not stopping
early. It is not a constant: measured here it falls from about 1.6-1.8 at
nseq <= 128 to 1.02 (network) and 1.07 (EM--BP) at nseq = 4096. That is this
project's standing warning -- the optimisation budget is a regularisation knob,
not a convergence detail -- turned into a number, and it says where the knob
matters: at small nseq, where pinning budgets instead of selecting them would
have distorted the comparison by up to two thirds.

It also explains why the at_cap slopes are steeper than the selected ones for
BOTH arms. Running to the cap overfits at small nseq and costs almost nothing at
large nseq, so the at-cap curve starts higher and ends level with the selected
one -- steepening it mechanically. An at-cap slope is therefore not an estimate
of any arm's convergence rate; it is useful here only because the DIFFERENCE
between two arms measured the same way remains meaningful.

WHERE THE GAP COMES FROM. Averaging over the noise schedule hides the most
informative structure in it. Fitted separately at each of the twelve noise
levels, the network's slope is nearly FLAT in t (-0.20 to -0.27) while EM-BP's
steepens monotonically (-0.31 at t=0.05 to -0.54 at t=3.0), so the gap tracks
log t with a correlation of -0.994. The whole t-dependence is one arm improving
faster with data as the noise rises.

That is the signature expected if the Markov factorisation is what pays. At small
t each site's own observation nearly determines its posterior and there is little
to pool across the chain, so knowing the structure buys little; at large t the
per-site signal is weak and pooling along the chain is the entire game. Note this
is a statement about ESTIMATION RATE and does not contradict the closure result,
which says the Gaussian-closure error shrinks at large t -- one is about how fast
a correctly specified estimator learns, the other about what a second-order
approximation discards.

Writes outputs/rate_robustness/{variants,budget_value,by_noise}.csv.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = "outputs/frozen"
OUT_DIR = "outputs/rate_robustness"

VARIANTS = {
    "selected": ("net_score_rel_l2_selected", "em_bp_score_rel_l2"),
    "oracle": ("net_score_rel_l2_oracle", "em_bp_score_rel_l2"),
    "at_cap": ("net_score_rel_l2_at_cap", "em_bp_score_rel_l2_at_cap"),
    "posterior_mean": ("net_mean_rel_l2_selected", "em_bp_mean_rel_l2"),
}

# The at-cap variant is the one that carries the argument, so it is the one with
# a gate. If removing budget selection entirely ever removes the effect, the
# finding was about selection and must be withdrawn rather than requalified.
AT_CAP_MIN_T = 2.0


def load():
    out = []
    for pat in (f"{ROOT}/exp_07_certified_seed*/sample_efficiency_val.csv",
                f"{ROOT}/exp_07_n4096_seed*/sample_efficiency_val.csv"):
        files = sorted(glob.glob(pat))
        if not files:
            print(f"REFUSING: no run at {pat}", file=sys.stderr)
            sys.exit(1)
        for f in files:
            seed = f.split("seed")[1].split("/")[0]
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    if int(r.get("em_resolved", 1)):   # same gate as the table
                        r["seed"] = seed
                        out.append(r)
    return out


def per_seed_by_size(rows, key):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(key, "")
        if v not in ("", None):
            acc[r["seed"]][int(r["n_chains"])].append(float(v))
    return {s: {n: float(np.mean(v)) for n, v in d.items()} for s, d in acc.items()}


def slopes(rows, key, sizes, seeds):
    tab = per_seed_by_size(rows, key)
    ln_n = np.log(np.array(sizes, dtype=float))
    out = []
    for s in seeds:
        if any(n not in tab[s] for n in sizes):
            return None
        out.append(np.polyfit(ln_n, np.log([tab[s][n] for n in sizes]), 1)[0])
    return np.array(out)


def se(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


def main():
    rows = load()
    seeds = sorted({r["seed"] for r in rows}, key=int)
    sizes = sorted({int(r["n_chains"]) for r in rows})
    os.makedirs(OUT_DIR, exist_ok=True)

    var_rows, at_cap_t = [], None
    for label, (nk, ek) in VARIANTS.items():
        ns, es = slopes(rows, nk, sizes, seeds), slopes(rows, ek, sizes, seeds)
        if ns is None or es is None:
            print(f"REFUSING: {label} has incomplete columns", file=sys.stderr)
            sys.exit(1)
        d = es - ns                                   # paired within seed
        t = float(np.mean(d) / se(d))
        if label == "at_cap":
            at_cap_t = t
        var_rows.append({
            "variant": label,
            "net_slope": float(np.mean(ns)), "net_se": se(ns),
            "em_slope": float(np.mean(es)), "em_se": se(es),
            "gap": float(np.mean(d)), "gap_se": se(d), "t": t,
            "em_steeper_seeds": int(np.sum(d < 0)), "n_seeds": len(seeds),
        })
        print(f"  {label:<15} net {np.mean(ns):+.4f}  em {np.mean(es):+.4f}  "
              f"gap {np.mean(d):+.4f}±{se(d):.4f}  t={t:6.2f}  "
              f"{int(np.sum(d < 0))}/{len(seeds)}")

    if at_cap_t is None or abs(at_cap_t) < AT_CAP_MIN_T:
        print(
            f"REFUSING: with budget selection removed entirely the gap is not "
            f"significant (t={at_cap_t}). Every objection to the rate finding is "
            f"about how a budget was chosen; if the effect needs a choice to "
            f"exist, it is a fact about selection and not about the methods.",
            file=sys.stderr,
        )
        sys.exit(1)

    # What a budget is worth, and how that decays.
    val_rows = []
    for arm, cap_key, sel_key in (
        ("network", "net_score_rel_l2_at_cap", "net_score_rel_l2_selected"),
        ("em_bp", "em_bp_score_rel_l2_at_cap", "em_bp_score_rel_l2"),
    ):
        cap, sel = per_seed_by_size(rows, cap_key), per_seed_by_size(rows, sel_key)
        for n in sizes:
            r = np.array([cap[s][n] / sel[s][n] for s in seeds])
            val_rows.append({
                "arm": arm, "n_chains": n,
                "at_cap_over_selected": float(np.mean(r)),
                "se": se(r),
                "overfit_penalty_pct": float(100.0 * (np.mean(r) - 1.0)),
            })

    # Per noise level. Seeds with any cell dropped by the resolution gate are
    # excluded for that t rather than averaged around: a slope fitted through a
    # missing size is not the same estimator as the others in the column.
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["seed"], float(r["t"]))][int(r["n_chains"])] = r
    ln_n = np.log(np.array(sizes, dtype=float))
    noise_rows = []
    for t in sorted({float(r["t"]) for r in rows}):
        ok = [s for s in seeds if all(n in idx[(s, t)] for n in sizes)]
        if len(ok) < 8:
            continue
        ns, es = [], []
        for s in ok:
            ns.append(np.polyfit(ln_n, np.log(
                [float(idx[(s, t)][n][VARIANTS["selected"][0]]) for n in sizes]), 1)[0])
            es.append(np.polyfit(ln_n, np.log(
                [float(idx[(s, t)][n][VARIANTS["selected"][1]]) for n in sizes]), 1)[0])
        ns, es = np.array(ns), np.array(es)
        d = es - ns
        noise_rows.append({
            "t": t, "n_seeds": len(ok),
            "net_slope": float(np.mean(ns)), "em_slope": float(np.mean(es)),
            "gap": float(np.mean(d)), "gap_se": se(d),
            "t_stat": float(np.mean(d) / se(d)),
            "em_steeper_seeds": int(np.sum(d < 0)),
        })
    if len(noise_rows) > 2:
        lt = np.log([r["t"] for r in noise_rows])
        gp = np.array([r["gap"] for r in noise_rows])
        print(f"  gap vs log t: r = {np.corrcoef(lt, gp)[0, 1]:+.3f}  "
              f"({gp.max():+.3f} at t={noise_rows[int(gp.argmax())]['t']:.3f} "
              f"to {gp.min():+.3f} at t={noise_rows[int(gp.argmin())]['t']:.3f})")

    for path, data in ((f"{OUT_DIR}/variants.csv", var_rows),
                       (f"{OUT_DIR}/budget_value.csv", val_rows),
                       (f"{OUT_DIR}/by_noise.csv", noise_rows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
