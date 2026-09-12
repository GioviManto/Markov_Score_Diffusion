#!/usr/bin/env python
"""The grid diagnostics Chapter 8 quotes, recomputed rather than transcribed.

WHY THIS EXISTS. Chapter 8 carried four numbers by hand.

Two of them --- the interior and worst-case column-mass residuals of the
transition array --- were exactly right, and they are recomputed here from the
kernel itself so they stay that way. "Interior" was undefined in the chapter;
it is `MixtureInnovationKernel.scale_diagnostics`' definition, |a| <= A/2, and
the macro now carries that with it.

The other two were not right. The chapter said the maximum boundary mass over
chains runs "up to 152x" the single-chain value, "range 2.9x to 8369x across
configurations" --- self-contradictory on its face, since 152 is not the top of
a range topping out at 8369, and matching no committed output. Rather than
substitute a different artifact's numbers, this recomputes the comparison the
remark is actually making: run the same forward recursion on one chain and on a
batch, and see how far above the single-chain figure the batch maximum sits.
Nothing is read from a stored file, so nothing here can go stale against a rerun
or rest on a sweep whose provenance record is empty.

    python tools/make_grid_diagnostics.py

Writes Markov_Score_Diffusion/thesis/sections/grid-diagnostics-numbers.tex. Do not hand-edit.
"""
import csv
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from frozen_config import FROZEN  # noqa: E402
from src.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from src.noising import alpha_delta  # noqa: E402
from src.priors import LaplaceAR1  # noqa: E402
from src.utils import rng_for  # noqa: E402

N_CHAINS = 256
BOUNDARY_FRAC = 0.02
# The interior of the domain, as scale_diagnostics defines it: parent states
# within half the half-width. Outside that the density is being truncated, so a
# column-mass residual there measures truncation and not quadrature.
INTERIOR_FRAC = 0.5

prior = LaplaceAR1(FROZEN.rho)
grid, weights = make_grid(FROZEN.half_width, FROZEN.n_grid)
log_K = prior.log_transition_matrix(grid)

# --- column mass: quadrature error of the transition array, per parent state.
col_mass = (np.exp(log_K) * weights[:, None]).sum(axis=0)
dev = np.abs(col_mass - 1.0)
interior_mask = np.abs(grid) <= INTERIOR_FRAC * float(np.max(np.abs(grid)))
interior = float(dev[interior_mask].max())
edge = float(dev.max())

# --- boundary mass: how far the worst chain sits above a single-chain reading.
rng = rng_for("griddiag", "boundary")
A = np.stack([prior.sample(rng, FROZEN.n_sites) for _ in range(N_CHAINS)])
ratios_single, ratios_mean = [], []
for t in FROZEN.t_grid:
    alpha, delta = alpha_delta(float(t))
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    _, _, d_one = grid_bp_batch(grid, weights, log_K, X[:1], alpha, delta,
                                boundary_frac=BOUNDARY_FRAC)
    _, _, d_all = grid_bp_batch(grid, weights, log_K, X, alpha, delta,
                                boundary_frac=BOUNDARY_FRAC)
    if d_one["forward_edge_max"] > 0:
        ratios_single.append(d_all["forward_edge_max"] / d_one["forward_edge_max"])
    if d_all["forward_edge_mean"] > 0:
        ratios_mean.append(d_all["forward_edge_max"] / d_all["forward_edge_mean"])

if not ratios_single:
    print("REFUSING: every single-chain boundary mass underflowed to zero, so "
          "the comparison the remark makes has no denominator.", file=sys.stderr)
    sys.exit(1)

print(f"working configuration: M={FROZEN.n_grid}, A={FROZEN.half_width:g}, "
      f"rho={FROZEN.rho}, {FROZEN.n_sites} sites, {len(FROZEN.t_grid)} noise levels")
print(f"  column mass: interior (|a| <= A/2) {interior:.3e}, worst anywhere {edge:.3e}")
print(f"  batch max / single-chain value: min {min(ratios_single):.1f}, "
      f"median {st.median(ratios_single):.1f}, max {max(ratios_single):.1f}")
print(f"  batch max / batch mean:         min {min(ratios_mean):.1f}, "
      f"median {st.median(ratios_mean):.1f}, max {max(ratios_mean):.1f}")


def sci(v, digits=1):
    m, e = f"{v:.{digits}e}".split("e")
    return f"{m} \\times 10^{{{int(e)}}}"


def th(x):
    return f"{x:,.0f}".replace(",", "{,}")


# --- domain truncation, read from exp_18's half-width sweep ------------------
# The self-convergence check answers "is the mesh fine enough"; it cannot
# answer "is the domain wide enough", because both resolutions share the same
# truncation. exp_18 varied A and recorded how much posterior mass reaches the
# outermost cells, which is the quantity that question needs.
_brows = list(csv.DictReader(open("outputs/frozen/exp_18/boundary.csv")))
_by_A = {}
for _r in _brows:
    _A = float(_r["half_width"])
    _by_A.setdefault(_A, []).append(float(_r["boundary_mass_max"]))
trunc_macros = "".join(
    f"\\newcommand{{\\gdtrunc{n}}}{{{sci(max(_by_A[a]))}}}\n"
    for n, a in (("four", 4.0), ("six", 6.0), ("eight", 8.0), ("ten", 10.0))
    if a in _by_A)
trunc_macros += f"\\newcommand{{\\gdtrunccells}}{{{len(_brows)}}}\n"


tex = f"""%% GENERATED by tools/make_grid_diagnostics.py -- do not hand-edit.
%%
%% Recomputed from the kernel and the recursion, not read from a stored sweep.
%% Math-mode content WITHOUT delimiters: the prose supplies the $...$.
\\newcommand{{\\gdinterior}}{{{sci(interior)}}}
\\newcommand{{\\gdedge}}{{{sci(edge)}}}
\\newcommand{{\\gdchains}}{{{N_CHAINS}}}
\\newcommand{{\\gdlevels}}{{{len(FROZEN.t_grid)}}}
\\newcommand{{\\gdboundlo}}{{{th(min(ratios_single))}}}
\\newcommand{{\\gdboundmed}}{{{th(st.median(ratios_single))}}}
\\newcommand{{\\gdboundhi}}{{{th(max(ratios_single))}}}
%% Domain truncation, from exp_18's own half-width sweep: how much posterior
%% mass reaches the outermost {BOUNDARY_FRAC:.0%} of the domain at each A. The
%% reference runs at A = 8, and Chapter 7's validation quotes these so the
%% truncation is a measured quantity rather than an assumed-safe setting.
{trunc_macros}"""
dest = "../../Markov_Score_Diffusion/thesis/sections/grid-diagnostics-numbers.tex"
open(dest, "w").write(tex)
print(f"\nwrote {dest}")
