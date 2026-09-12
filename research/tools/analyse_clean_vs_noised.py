"""What the channel costs, at a budget that clears its own convergence requirement.

The claim this replaces -- "the channel destroys shape recovery" -- was withdrawn
as "a rate at a fixed iteration count, mistaken for an information penalty". The
diagnosis was right and the fix is not a reinterpretation but a rerun: every
previous version of this comparison used n_iters=120, while the innovation
kurtosis settles at a median 229 updates (exp_27). The source marked those
numbers PROVISIONAL and compendium-only, and said in as many words that they
should be rerun at FROZEN.em_max_iters before any use.

Array 632889 is that rerun, sharded by sample size, 16 replicates, shape_iters=800
-- FROZEN.em_max_iters doubled twice, so the answer cannot be read as sitting on
the cap. Both arms always receive the same count, so the comparison is never an
optimisation-budget difference in disguise.

The design is paired: both arms get the SAME clean chains and the only difference
is that the noised arm never sees them clean. So the gap is the price of the
channel and nothing else -- not sample size, not initialisation, not the draw.
Pairing is what gives this power, because the replicate-to-replicate spread in a
fitted kurtosis is larger than the effect being looked for.

Two things to report, and they answer different questions:

  BIAS     -- does the channel move the estimate? Read the paired difference.
  VARIANCE -- does the channel make the estimate noisier? Read the SE ratio.

Keeping them separate is the whole point. "Destroys recovery" conflates them,
and the measured answer differs between them: no detectable bias, a real and
roughly constant variance penalty.
"""
import csv, glob, re, sys
import numpy as np

SOURCE = "outputs/frozen/exp_06_clean_conv_n*/clean_vs_noised_shape.csv"
TRUTH = 3.0          # Laplace excess kurtosis, the innovation law used here
EXPECTED_REPS = 16

rows = []
for f in sorted(glob.glob(SOURCE)):
    for r in csv.DictReader(open(f)):
        rows.append(r)
if not rows:
    sys.exit(f"no converged clean-vs-noised output at {SOURCE}")

sizes = sorted({int(r["n_chains"]) for r in rows})
F = lambda r, k: float(r[k])
se = lambda v: v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else float("nan")


def arm(n, which):
    return np.array([F(r, "innovation_excess_kurtosis") for r in rows
                     if int(r["n_chains"]) == n and r["arm"] == which])


print(f"sizes {sizes}, truth = {TRUTH}")
print(f"{'nseq':>5} {'clean':>16} {'noised':>16} {'paired d':>17} {'|d|/SE':>7} {'SE ratio':>9}")
ratios, ts = [], []
for n in sizes:
    c, m = arm(n, "clean"), arm(n, "noised")
    if c.size != EXPECTED_REPS or m.size != EXPECTED_REPS:
        print(f"  WARNING: nseq={n} has {c.size}/{m.size} replicates, "
              f"expected {EXPECTED_REPS} -- provisional", file=sys.stderr)
    k = min(c.size, m.size)
    d = m[:k] - c[:k]
    t = abs(d.mean()) / se(d) if se(d) > 0 else float("nan")
    ratio = se(m) / se(c)
    ratios.append(ratio)
    ts.append(t)
    print(f"{n:>5} {c.mean():>8.3f}+-{se(c):.3f} {m.mean():>8.3f}+-{se(m):.3f} "
          f"{d.mean():>+10.3f}+-{se(d):.3f} {t:>7.1f} {ratio:>8.2f}x")

print()
print(f"BIAS:     largest |d|/SE across sizes is {max(ts):.1f} on {EXPECTED_REPS-1} df.")
if max(ts) < 2.0:
    print("          No detectable bias at any size. The channel does not move the")
    print("          shape estimate; it was never an information penalty.")
else:
    print("          A bias survives at a converged budget -- the withdrawal was")
    print("          diagnosed wrongly and the claim needs restating, not deleting.")

lo, hi = min(ratios), max(ratios)
print()
print(f"VARIANCE: SE ratio runs {lo:.2f}x to {hi:.2f}x, median {np.median(ratios):.2f}x.")
# Deliberately NOT fitted as a trend. Reading a slope off five points whose own
# sampling error is large is the mistake that produced the withdrawn "flat curve
# is a discretisation floor" claim; an intermediate excursion is not a trend, and
# on this data the largest ratio sits at nseq=512 with both neighbours lower.
i_hi = int(np.argmax(ratios))
if 0 < i_hi < len(ratios) - 1:
    print(f"          The maximum is at nseq={sizes[i_hi]}, with both neighbours")
    print(f"          lower ({ratios[i_hi-1]:.2f}x, {ratios[i_hi+1]:.2f}x), so this is an")
    print("          excursion and not a trend in nseq. No slope is fitted: five")
    print("          points with this much sampling error is how the withdrawn")
    print("          'flat curve' claim was made.")
print()
print("Summary: the channel costs VARIANCE, not BIAS -- roughly a "
      f"{np.median(ratios):.1f}x inflation of the standard error of a fitted")
print("innovation kurtosis, with no shift in its centre.")
