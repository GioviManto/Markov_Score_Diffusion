"""Guards the returned-iterate/trace alignment of exp_08's gradient ascent.

The failure this pins down is silent and produces a plausible-looking table. The
old loop appended `(theta, logL)` at the top of the body, *then* stepped, and
returned the stepped parameter. So the returned kernel was one update beyond
anything that had been evaluated, and each result row mixed two different
models: `param0_err`/`param1_err` measured the returned parameter, while
`logL_final`, `logL_gap_to_em` and `monotone_violation` all described its
predecessor. Nothing crashes, no number looks absurd, and the discrepancy shrinks
as the run converges -- which is exactly why it survived review.

`fit_em` in src/em.py had the same bug and was already repaired; these tests hold
the experiment side to the same invariant.

Run:  python -m pytest tests/test_gradient_trace_alignment.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from exp_08_gradient_vs_exact_mstep import (  # noqa: E402
    gradient_ascent,
    noisy_groups,
)

from src.bp_grid import make_grid  # noqa: E402
from src.em import e_step_multi  # noqa: E402
from src.kernels import GaussianAR1Kernel  # noqa: E402
from src.priors import GaussianAR1  # noqa: E402
from src.utils import rng_for  # noqa: E402

N_SITES = 8
T_TRAIN = (0.2, 0.8)


def clip_gauss(v):
    """Same admissible-set projection exp_08's `main` uses for the Gaussian arm."""
    return GaussianAR1Kernel(
        float(np.clip(v[0], -0.99, 0.99)),
        float(np.clip(v[1], 1e-3, 10.0)),
    )


def _setup(n_chains=6, n_grid=161):
    grid, weights = make_grid(8.0, n_grid)
    rng = rng_for("test-grad-trace")
    prior = GaussianAR1(0.8)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    groups = noisy_groups(A, T_TRAIN, rng)
    return grid, weights, groups


def test_returned_kernel_matches_final_trace_entry():
    """The returned theta is the one the last trace entry was evaluated at."""
    grid, weights, groups = _setup()
    n_updates = 5

    fitted, trace = gradient_ascent(
        GaussianAR1Kernel(0.3, 0.8), grid, weights, groups,
        lr=0.5, n_iters=n_updates, project=clip_gauss,
    )

    np.testing.assert_allclose(
        np.asarray(fitted.theta, dtype=float),
        trace["theta"][-1],
        rtol=0, atol=0,
    )


def test_final_log_evidence_is_the_returned_kernels_own():
    """Recomputing the evidence at the returned kernel reproduces the trace tail.

    This is the assertion the old code could not satisfy: its `logL_final`
    belonged to the previous iterate.
    """
    grid, weights, groups = _setup()

    fitted, trace = gradient_ascent(
        GaussianAR1Kernel(0.3, 0.8), grid, weights, groups,
        lr=0.5, n_iters=5, project=clip_gauss,
    )

    independent = e_step_multi(
        grid, weights, fitted.log_transition_matrix(grid), groups
    ).log_evidence

    assert abs(independent - trace["log_evidence"][-1]) <= 1e-9 * abs(independent)


def test_trace_has_one_more_state_than_updates():
    """n_updates + 1 evaluated states, and n_updates is stored, not inferred."""
    grid, weights, groups = _setup()

    for n_updates in (1, 4, 7):
        _, trace = gradient_ascent(
            GaussianAR1Kernel(0.3, 0.8), grid, weights, groups,
            lr=0.5, n_iters=n_updates, project=clip_gauss,
        )
        assert trace["n_updates"] == n_updates
        assert len(trace["log_evidence"]) == n_updates + 1
        assert len(trace["theta"]) == n_updates + 1


def test_monotonicity_check_covers_the_final_update():
    """A descent on the *last* update must be visible in the trace.

    This is the sharp version of the bug. At lr = 10 this configuration
    overshoots and the second update decreases the evidence. With n_iters = 2
    that descent is the final step, so the old alignment -- which scored only
    the states before each update -- recorded increments of length 1 and
    reported the run as perfectly monotone. The repaired loop scores both
    updates and the violation is visible.

    A much larger lr does not work as a probe: `clip_gauss` pins the iterate to
    the boundary, every later increment is exactly 0.0, and the run looks
    monotone for a genuine reason.
    """
    grid, weights, groups = _setup()

    _, trace = gradient_ascent(
        GaussianAR1Kernel(0.3, 0.8), grid, weights, groups,
        lr=10.0, n_iters=2, project=clip_gauss,
    )

    d = np.diff(np.asarray(trace["log_evidence"], dtype=float))
    assert d.size == 2, "every update must contribute a scored increment"
    assert d[-1] < 0.0, "the overshoot on the final update must be scored"
