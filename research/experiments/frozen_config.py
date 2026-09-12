"""The frozen configuration behind every number in the paper.

Why this file exists
--------------------
The previous draft stitched its results together from runs that did not agree
with each other: EM stopped at 40, 120 or 200 iterations depending on the
experiment, replicate counts were 3, 6 or 16, and the autoregressive truth was
0.8 in some experiments and 0.85 in others. Numbers from those runs could not
be put in the same table, several conclusions turned out to be artefacts of the
iteration budget rather than of the model, and the paper ended up carrying its
own corrections.

So: one configuration, used everywhere, defined once, imported. An experiment
that needs a different setting must say so explicitly at the call site, and
that difference then has to be reported. Nothing in the paper may come from a
run that silently diverged from this file.

Everything the paper reports is produced under `FROZEN` and written beneath
`outputs/frozen/`.

Usage
-----
    from frozen_config import FROZEN, frozen_settings, provenance

    settings = frozen_settings(sizes=(32, 128, 512))   # narrow, never widen
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
FROZEN_ROOT = PACKAGE_ROOT / "outputs" / "frozen"


def _log_grid(lo: float, hi: float, n: int) -> tuple[float, ...]:
    """`n` points log-spaced on [lo, hi], rounded so they print readably."""
    pts = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    return tuple(round(float(p), 4) for p in pts)


@dataclass(frozen=True)
class FrozenConfig:
    """One configuration. Frozen dataclass: mutation is a bug, not an override."""

    # --- the chain -------------------------------------------------------
    rho: float = 0.85
    """Autoregressive truth. One value everywhere. The old 0.8 runs are retired:
    two values meant two populations of results that could never be pooled."""

    n_sites: int = 32
    """Chain length L, which is also the ambient dimension."""

    innovation: str = "laplace"
    """Default innovation family. Excess kurtosis 3, symmetric, so the third
    cumulant vanishes and the fourth is the leading non-Gaussian coordinate."""

    # --- the quadrature grid ---------------------------------------------
    n_grid: int = 401
    half_width: float = 8.0
    """Chosen from the grid/domain sweep. At (401, 8), over the twelve-level
    schedule with 256 chains per cell, the forward-message boundary mass is

        worst chain   1.2e-6      p90   1.5e-8      median   3.0e-9

    and the interior column-mass residual is 9.6e-4. The conclusion stands --
    all of these are far below every effect reported -- but note WHICH statistic
    is quoted. This docstring used to say "the truncation residual is 1.4e-8",
    which is the p90, not a bound: the worst chain is 84x larger. A maximum over
    sites of a sampled trajectory is a statistic, and quoting an upper quantile
    of it as though it bounded the error is the mistake the boundary diagnostic
    was rewritten to stop making. Regenerate with

        python experiments/exp_18_revision_diagnostics.py --parts boundary \\
            --out outputs/frozen/exp_18

    See Rung 0."""

    # --- the noise schedule ----------------------------------------------
    t_grid: tuple[float, ...] = field(
        default_factory=lambda: _log_grid(0.05, 3.0, 12)
    )
    """Twelve log-spaced levels on [0.05, 3.0]. The old six-point grid was too
    coarse to fit a decay slope: over most of it only two points survived above
    the noise floor, so a rate could be asserted but not measured."""

    # --- replication ------------------------------------------------------
    n_seeds: int = 16
    """Sixteen everywhere. At three, an RMSE is estimated from three numbers and
    its own sampling error is comparable to the effects being read off it --
    which is how a curve got read as flat when it was falling like M^{-1/2}."""

    sizes: tuple[int, ...] = (32, 128, 256, 512, 1024, 2048, 4096)
    """Training-set sizes M, seven values."""

    efficiency_sizes: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048)
    """The sizes the network comparison uses, which are NOT `sizes`.

    Two deliberate differences, recorded here because the experiment used to
    carry this list inline and the paper's protocol appendix quoted a third
    list that matched neither -- the exact drift this file exists to stop.

    64 is added: the ratio moves fastest at the small end, and 32 -> 128 is one
    jump across the region where the two arms separate.

    4096 was dropped, on the argument that at 2048 both arms already select the
    largest budget the checkpoint grid offers in 33% and 61% of cells, so that
    row is bounded by the budget rather than by the data and a further doubling
    would only add a second such row. That argument was wrong. Selecting the cap
    means validation error is still falling, not that it is falling fast enough
    to matter; tripling both caps at 2048 (job 631467, 16 seeds, paired) moves
    the ratio by -0.16 +/- 0.18. 4096 was then run separately at a raised budget
    (jobs 631496/631497, 16 seeds on H200) and IS in the table.

    This list stays at seven sizes regardless, because it is the protocol the
    certified outputs were produced under and widening it would silently
    invalidate their provenance. The 4096 row has its own source directory and
    the generator joins the two, with the budget difference disclosed in the
    caption and calibrated by the 2048 rerun."""

    n_heldout: int = 256
    heldout_seed_offset: int = 1_000_000
    """Held-out sequences are drawn from a seed disjoint from every training
    seed, and the offset is large enough that no training seed can reach it."""

    # --- EM stopping ------------------------------------------------------
    em_max_iters: int = 400
    em_loglik_tol: float = 1e-9
    em_shape_tol: float = 1e-3
    """Stop on the SHAPE, never on rho.

    The coordinates converge at very different rates. At rho=0.85, M=512, C=8
    the autoregressive coefficient is flat from iteration ~25, while the fitted
    innovation excess kurtosis reads 0.84 at 30 iterations, 1.85 at 60, 2.30 at
    120 and 2.29 at 400. Monitoring rho -- the natural thing to plot -- certifies
    a kernel that still carries a third of the true higher-order structure, and
    every conclusion drawn at 40 iterations was drawn on an underfitted kernel.

    So the rule is: stop when the fitted excess kurtosis stops moving, and treat
    the log-likelihood tolerance as a secondary guard. Both must be satisfied."""

    n_components: int = 8
    """Innovation mixture size. Eight is the paired-design optimum: 16 differs
    from it by less than one standard error and at M=128 is marginally worse."""

    # --- the ring model (Rung 4) -----------------------------------------
    ring_n_frames: int = 12
    """T; ambient dimension is D = 2T = 24."""
    ring_sigma: float = 0.30
    ring_lambda: float = 0.05
    """Width of the quadratic confining well V(r) = (r - r_star)^2 / 2 lam.
    Estimated at Rung 4a, together with r_star."""
    ring_r_star: float = 1.0
    """Radius the well sits at."""
    ring_psi_true: float = np.pi / 6
    ring_n_psi: int = 128
    """Angular grid for the p(psi) E-step."""

    # --- misc -------------------------------------------------------------
    base_seed: int = 20260816

    def seeds(self) -> tuple[int, ...]:
        """The replicate seeds. Derived from `base_seed` so a run is reproducible
        from this file alone, and mixed so that consecutive seeds do not produce
        correlated streams."""
        rng = np.random.default_rng(self.base_seed)
        return tuple(int(s) for s in rng.integers(0, 2**31 - 1, size=self.n_seeds))

    def heldout_seed(self) -> int:
        return self.base_seed + self.heldout_seed_offset

    @property
    def innovation_variance(self) -> float:
        """q = 1 - rho^2, which makes Var(a_i) = 1 at every site and therefore
        makes every innovation family share Cov(a_i, a_j) = rho^{|i-j|} exactly.
        That is what turns the family into a controlled probe: the families
        differ only beyond second moments."""
        return 1.0 - self.rho**2

    def grid(self) -> np.ndarray:
        return np.linspace(-self.half_width, self.half_width, self.n_grid)

    def output_dir(self, name: str) -> Path:
        return FROZEN_ROOT / name

    def as_dict(self) -> dict:
        return asdict(self)

    def narrowed(self, **kwargs) -> "FrozenConfig":
        """A copy with some fields changed.

        For deliberate, reported deviations only -- a smoke test, or an
        experiment that genuinely needs a shorter chain. The result is still a
        frozen dataclass, and `provenance()` records the difference, so a
        narrowed run cannot be mistaken for a frozen one.
        """
        return replace(self, **kwargs)


FROZEN = FrozenConfig()


def frozen_settings(**overrides) -> dict:
    """The frozen config as a plain settings dict, for `common.apply_overrides`."""
    settings = FROZEN.as_dict()
    unknown = set(overrides) - set(settings)
    if unknown:
        raise SystemExit(f"frozen_settings: unknown keys {sorted(unknown)}")
    settings.update(overrides)
    return settings


def config_hash(config: FrozenConfig = FROZEN) -> str:
    """A short stable digest of the whole configuration.

    The full dict is already written into every params.json, which is what you
    need to reproduce a run. What it does not give you is a way to ask "did
    these two outputs come from the same configuration?" without diffing two
    dictionaries and knowing which key differences matter. A digest turns that
    into a string comparison, so a merge step can refuse mismatched shards
    instead of averaging them.

    Sorted keys and `repr` so the digest depends on the values and not on field
    declaration order; tuples and floats both round-trip through `repr` exactly.
    """
    import hashlib

    payload = "\n".join(
        f"{k}={v!r}" for k, v in sorted(config.as_dict().items())
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def provenance(config: FrozenConfig = FROZEN) -> dict:
    """What to write into every params.json, so an output file says which
    configuration produced it and whether that was the frozen one."""
    diff = {
        k: v for k, v in config.as_dict().items()
        if v != FROZEN.as_dict().get(k)
    }
    return {
        "config": config.as_dict(),
        "config_hash": config_hash(config),
        "frozen_hash": config_hash(FROZEN),
        "is_frozen": not diff,
        "deviations_from_frozen": diff,
    }


__all__ = [
    "FROZEN",
    "FrozenConfig",
    "FROZEN_ROOT",
    "frozen_settings",
    "provenance",
    "config_hash",
]
