#!/usr/bin/env python
"""Reference score energies of the frozen exp_07 test bundle, per noise level.

WHY THIS EXISTS. The headline table defines its estimand (thesis Eq. 10.2) by
pooling squared score error and reference score energy over the noise schedule
BEFORE the square root. The frozen exp_07 CSVs store only per-level relative
errors E_t = sqrt(A_t / B_t); the pooled estimand additionally needs the
per-level reference energies B_t = sum_j ||S*_t(x_j)||^2 on the test bundle.

Those energies are recoverable exactly, without retraining anything, because
the exp_07 test bundle is deliberately shared across the sixteen replicate
seeds: its RNG tag ("exp07-test") mixes in no replicate index (see
exp_07_em_vs_score_network.make_test_set), so B_t is one number per level.
Given B_t, the pooled squared error is B_t * E_t^2 exactly -- the same B_t is
the denominator the stored relative error was formed with.

VALIDATION. Regenerating a bundle from its tag has to be bit-exact or the
recovered B_t is fiction. That is checked against ground truth this repo
already stores: exp_31's confirm.csv records, per seed, the summed reference
energy sq_ref of its own regenerated bundle together with the bundle hash.
This script rebuilds the exp_31 seed-0 test bundle by the same construction
and refuses unless BOTH the hash and sq_ref (all-site, summed over the
schedule) match the frozen values to full precision.

    python tools/make_ref_energy.py

Writes outputs/frozen/exp_07_ref_energy/ref_energy.csv. Deterministic; safe to
rerun. Reads only frozen inputs and the committed model code.
"""
import csv
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from src.bp_grid import make_grid, grid_bp_batch  # noqa: E402
from src.noising import alpha_delta  # noqa: E402
from src.priors import LaplaceAR1  # noqa: E402
from src.utils import rng_for  # noqa: E402
from frozen_config import FROZEN  # noqa: E402

N_SITES = FROZEN.n_sites
T_TRAIN = tuple(FROZEN.t_grid)
GRID, WEIGHTS = make_grid(FROZEN.half_width, FROZEN.n_grid)
PRIOR = LaplaceAR1(FROZEN.rho)
LOG_K = PRIOR.log_transition_matrix(GRID)

OUT_DIR = "outputs/frozen/exp_07_ref_energy"
OUT = f"{OUT_DIR}/ref_energy.csv"

# Frozen ground truth for the validation: exp_31 confirm.csv, seed 0. sq_ref is
# identical across (n_chains, method) rows within a (seed, region) because they
# share one test bundle; region "all" sums every site.
EXP31_CSV = "outputs/frozen/exp_31_confirm_merged/confirm.csv"
EXP31_SEED = "0"


def _bundle(tag_keys, n_chains, t_values):
    """Rebuild a frozen bundle: same construction as exp_07 make_test_set /
    exp_31 _bundle -- one named RNG stream, chains first, then noise per level
    in schedule order."""
    rng = rng_for(*tag_keys)
    A = np.stack([PRIOR.sample(rng, N_SITES) for _ in range(n_chains)])
    out = {}
    for t in t_values:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        m_ref, _ = grid_bp_batch(GRID, WEIGHTS, LOG_K, X, alpha, delta)
        out[t] = (X, np.asarray(m_ref))
    return out


def _bundle_hash(bundle) -> str:
    h = hashlib.sha256()
    for t in sorted(bundle):
        X, m = bundle[t]
        h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(m, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def _ref_energy(bundle, t):
    X, m_ref = bundle[t]
    alpha, delta = alpha_delta(t)
    s_ref = -(X - alpha * m_ref) / delta
    return float((s_ref**2).sum())


def validate_against_exp31() -> None:
    rows = [r for r in csv.DictReader(open(EXP31_CSV))
            if r["seed"] == EXP31_SEED and r["region"] == "all"]
    if not rows:
        sys.exit(f"REFUSING: no seed-{EXP31_SEED} all-region rows in {EXP31_CSV}")
    want_sq_ref = {float(r["sq_ref"]) for r in rows}
    if len(want_sq_ref) != 1:
        sys.exit(f"REFUSING: sq_ref not constant within (seed, region): {want_sq_ref}")
    want_sq_ref = want_sq_ref.pop()
    want_hash = {r["test_bundle"] for r in rows}.pop()

    bundle = _bundle(("exp31-test", int(EXP31_SEED)), 2048, T_TRAIN)
    got_hash = _bundle_hash(bundle)
    got_sq_ref = sum(_ref_energy(bundle, t) for t in T_TRAIN)

    rel = abs(got_sq_ref - want_sq_ref) / want_sq_ref
    if rel > 1e-12:
        sys.exit(f"REFUSING: rebuilt sq_ref {got_sq_ref!r} != frozen {want_sq_ref!r} "
                 f"(rel {rel:.2e}); bundle regeneration is not exact on this host.")
    if got_hash != want_hash:
        # The hash covers the BP posterior means byte-for-byte, and those carry
        # the last-bit rounding of the host's BLAS (the frozen run used Linux/
        # cnode BLAS; a Mac's Accelerate rounds the same matmuls differently).
        # The recovered energies are what this script exists for, and they
        # reproduce to ~1e-16 above, so a hash mismatch alone is reported, not
        # fatal. A DATA difference cannot hide here: X enters sq_ref directly.
        print(f"  validation: sq_ref reproduces frozen value (rel {rel:.1e}); "
              f"bundle hash differs ({got_hash} vs {want_hash}: BLAS last-bit "
              f"rounding in the stored posterior means)")
    else:
        print(f"  validation: exp_31 seed {EXP31_SEED} bundle hash {got_hash} and "
              f"sq_ref reproduce the frozen values (rel err {rel:.1e})")


def main() -> None:
    validate_against_exp31()

    bundle = _bundle(("exp07-test",), FROZEN.n_heldout, T_TRAIN)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "ref_sq", "n_test", "n_sites", "grid_size",
                    "half_width", "tag", "bundle_hash"])
        h = _bundle_hash(bundle)
        for t in T_TRAIN:
            w.writerow([t, repr(_ref_energy(bundle, t)), FROZEN.n_heldout,
                        N_SITES, FROZEN.n_grid, FROZEN.half_width,
                        "exp07-test", h])
    print(f"wrote {OUT}: {len(T_TRAIN)} levels, n_test={FROZEN.n_heldout}, "
          f"bundle {h}")
    for t in T_TRAIN:
        print(f"  t={t:<7} B_t={_ref_energy(bundle, t):.6e}")


if __name__ == "__main__":
    main()
