"""One profile function, used for real images and generated images alike.

Keeping this in a single place is not tidiness: the comparison in `exp_25` is
between a set of real held-out images and a set of generated ones, and if the two
went through even slightly different code -- a different quantile convention, a
different standardisation -- the difference would be partly an artefact of the
measurement. Every number below is computed by the same function on both sides.
"""

from __future__ import annotations

import numpy as np

from .hierarchy import TreeIndex
from .sample_metrics import excess_kurtosis
from .scale_kernel import linear_ar_magnitude_ratio
from .wavelet import ORIENTATIONS, images_to_tree


def subband_profile(images: np.ndarray, levels: int) -> list[dict]:
    """Per-(orientation, depth) scale and excess kurtosis."""
    qt, nodes, scaling = images_to_tree(images, levels)
    depth_of = qt.node_depth
    rows = []
    for oi, orient in enumerate(ORIENTATIONS):
        for d in range(qt.depth + 1):
            block = nodes[:, oi, depth_of == d]
            rows.append({
                "orientation": orient,
                "tree_depth": d,
                "std": float(block.std()),
                "excess_kurtosis": excess_kurtosis(block),
            })
    rows.append({
        "orientation": "LL", "tree_depth": -1,
        "std": float(scaling.std()), "excess_kurtosis": excess_kurtosis(scaling),
    })
    return rows


def crossscale_profile(images: np.ndarray, levels: int) -> list[dict]:
    """Per-boundary linear correlation, magnitude ratio, and excess over the null."""
    qt, nodes, _ = images_to_tree(images, levels)
    TreeIndex(depth=qt.depth, branching=4)
    depth_of = qt.node_depth
    rows = []
    for oi, orient in enumerate(ORIENTATIONS):
        for d in range(qt.depth):
            parent = np.repeat(nodes[:, oi, depth_of == d], 4, axis=1).ravel()
            child = nodes[:, oi, depth_of == d + 1].ravel()
            ps, cs = parent / parent.std(), child / child.std()
            rho = float(np.corrcoef(ps, cs)[0, 1])
            q = np.quantile(np.abs(ps), [0.25, 0.75])
            lo = float(cs[np.abs(ps) <= q[0]].std())
            hi = float(cs[np.abs(ps) >= q[1]].std())
            ratio = hi / lo if lo > 0 else float("nan")
            null = linear_ar_magnitude_ratio(rho)
            rows.append({
                "orientation": orient,
                "parent_depth": d,
                "corr_linear": rho,
                "std_ratio_q4_q1": ratio,
                "std_ratio_linear_ar_null": null,
                "std_ratio_excess_over_null": ratio - null,
            })
    return rows


def profile_gap(generated: list[dict], real: list[dict], key: str) -> dict:
    """Mean and worst gap in `key` between two profiles, row-aligned.

    Returns the *signed* mean as well, because a model that is uniformly too
    light-tailed and one that scatters either side of the truth are different
    failures and an absolute summary hides that.

    **Relative gaps are reported alongside the absolute ones, and for `std` they
    are the ones to read.** Subband standard deviations span 0.14 to 17 in a
    natural image, so an absolute summary is simply a report about the LL band:
    a 1.4 absolute gap there is a 9 % error, while the same number at the finest
    subband would be a factor of four. Reading `worst_abs_gap` on `std` cost me a
    wrong conclusion once -- a model whose *worst* absolute gap grew while every
    detail band improved -- so both are returned and the ambiguity is removed at
    the source.
    """
    g = np.array([r[key] for r in generated], dtype=float)
    r = np.array([row[key] for row in real], dtype=float)
    if g.shape != r.shape:
        raise ValueError("profiles are not row-aligned")
    finite = np.isfinite(g - r)
    g, r = g[finite], r[finite]
    d = g - r
    scale = np.maximum(np.abs(r), 1e-12)
    rel = d / scale
    return {
        "mean_abs_gap": float(np.mean(np.abs(d))),
        "mean_signed_gap": float(np.mean(d)),
        "worst_abs_gap": float(np.max(np.abs(d))),
        "mean_rel_gap": float(np.mean(np.abs(rel))),
        "worst_rel_gap": float(np.max(np.abs(rel))),
    }
