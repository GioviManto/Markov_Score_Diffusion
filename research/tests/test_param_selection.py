"""Parameterisation selection must be deterministic across processes.

This project has now hit the same failure twice: a result that depends on CPython's
per-process `hash` salt for strings. First in `exp_18`'s seeding, where
`abs(hash(tag)) % 2**32` made committed diagnostics irreproducible from a fresh
interpreter. Then in `_selector`'s global-mode majority vote, where
`max(set(votes), key=votes.count)` resolved a tie by set iteration order.

Both are invisible under the default configuration -- five probe levels and two
parameterisations cannot tie -- and both appear the moment someone passes
`--set t_probe=(...)` with an even number of levels, which the experiment explicitly
supports for cluster sweeps. Hence a test rather than a comment.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

sys.path.insert(0, "experiments")

from exp_16_sampling_validation import _selector  # noqa: E402


def _best(mapping):
    return {("mlp", t): v for t, v in mapping.items()}


def test_global_selection_picks_the_majority():
    best = _best({0.1: "eps", 0.2: "eps", 0.4: "eps", 0.8: "x0", 1.6: "x0"})
    fn = _selector(best, "mlp", (0.1, 0.2, 0.4, 0.8, 1.6), "global")
    assert {fn(t) for t in (0.02, 0.5, 3.0)} == {"eps"}


def test_global_selection_is_constant_across_the_whole_trajectory():
    """The point of global mode: one score model for the entire integration.

    Per-level switching hands the integrator a field with jump discontinuities in t between
    two independently trained networks, so the generated sample reflects an estimator that
    was never fitted.
    """
    best = _best({0.1: "eps", 0.2: "x0", 0.4: "eps", 0.8: "x0", 1.6: "eps"})
    fn = _selector(best, "mlp", (0.1, 0.2, 0.4, 0.8, 1.6), "global")
    assert len({fn(t) for t in (0.02, 0.1, 0.3, 0.9, 2.0, 3.0)}) == 1

    per_level = _selector(best, "mlp", (0.1, 0.2, 0.4, 0.8, 1.6), "per_level")
    assert len({per_level(t) for t in (0.1, 0.2, 0.4, 0.8, 1.6)}) == 2


def test_tie_is_broken_deterministically_within_one_process():
    """An even split must resolve the same way every call, and lexicographically."""
    best = _best({0.1: "eps", 0.2: "eps", 0.4: "x0", 0.8: "x0"})
    probe = (0.1, 0.2, 0.4, 0.8)
    picks = {_selector(best, "mlp", probe, "global")(0.5) for _ in range(50)}
    assert picks == {"eps"}


def test_tie_is_broken_identically_across_hash_seeds():
    """The regression test proper: run the tie-break in fresh interpreters with different
    PYTHONHASHSEED values and require identical answers.

    `max(set(votes), key=votes.count)` fails this -- measured returns of eps/x0/x0/eps/eps
    across seeds 1..5 -- while sorting by (-count, value) does not. A subprocess is the only
    way to test it, because the salt is fixed once at interpreter start.
    """
    snippet = (
        "votes=['eps','eps','x0','x0'];"
        "print(sorted(set(votes), key=lambda v: (-votes.count(v), v))[0])"
    )
    results = set()
    for seed in ("0", "1", "2", "3", "42", "999"):
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        results.add(out.stdout.strip())
    assert results == {"eps"}, f"tie-break varied across hash seeds: {results}"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        _selector(_best({0.1: "eps"}), "mlp", (0.1,), "whatever")
