"""Which modules `BP_DEVICE` actually reaches.

`test_backend_parity` checks that the CPU and GPU paths AGREE. It cannot check
that a given module has a GPU path at all: a module that ignores `backend`
entirely agrees with itself perfectly and passes every parity test ever written.

That gap was live for a month. `backend` was wired into `bp_grid` and `denoiser`
on 19 Aug 2026, but `src/em.py` keeps its own copy of the forward-backward
recursion in `_e_step_chunk` and stayed pure numpy -- so `BP_DEVICE=gpu`
accelerated the grid-BP reference and the network arm while the EM fit, which
dominates the cost at high `em_iters` and is the arm the headline claim rests
on, ran on the CPU. Jobs 631496/631497 were sent to H200 nodes on the
understanding that they were GPU runs; they were GPU runs for part of their
work. Nothing failed, nothing warned, and the parity suite was green throughout.

These tests need no GPU, which is the point: the failure they catch is invisible
on the cluster (where everything merely runs slowly) and detectable anywhere.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Modules whose inner loops are the compute. If one of these stops importing
# `backend`, a device run silently becomes a CPU run wearing a GPU job's name.
#
# The list is deliberately about ARITHMETIC, not about tidiness. `kernels.py` is
# absent on purpose: the M-step is a closed-form update on (M, M) sufficient
# statistics that the E-step already reduced, it is not where the time goes, and
# the device boundary is drawn inside `_e_step_chunk` precisely so the M-step can
# stay numpy-only. Adding it here would be asking for a port nobody wants.
HOT_MODULES = (
    "bp_grid.py",       # grid BP message passing
    "denoiser.py",      # DSM network training and inference
    "em.py",            # EM E-step: the dominant cost of every fit
    "wavelet_bp.py",    # quadtree recursion for the image/video work
)


def _imports_backend(path: pathlib.Path) -> bool:
    """True if the module imports anything from `src.backend`.

    Parsed rather than grepped so that the word appearing in a docstring or a
    comment -- which is how this file itself would match -- does not count.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("src.backend", "backend"):
            return True
        if isinstance(node, ast.Import):
            if any(a.name in ("src.backend", "backend") for a in node.names):
                return True
    return False


@pytest.mark.parametrize("module", HOT_MODULES)
def test_hot_module_is_backend_aware(module):
    path = SRC / module
    assert path.exists(), f"{module} has moved; update HOT_MODULES"
    assert _imports_backend(path), (
        f"{module} does not import src.backend, so BP_DEVICE cannot reach it. "
        f"Any GPU job whose cost is in this module is really a CPU job. If the "
        f"module genuinely has no arithmetic worth putting on a device, remove "
        f"it from HOT_MODULES with a note saying why -- do not leave it silent."
    )


def test_em_e_step_routes_through_get_xp():
    """`em` must not merely import the backend -- it must use it.

    Importing `get_xp` and then calling `np.` throughout would satisfy the test
    above while restoring the exact bug. This asserts the module actually calls
    the selector.
    """
    tree = ast.parse((SRC / "em.py").read_text())
    calls = {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "get_xp" in calls, "src/em.py imports get_xp but never calls it"


def test_e_step_returns_host_arrays_by_construction():
    """The device boundary is inside the E-step, and its exit must convert.

    Every consumer of `ExpectedStatistics` is numpy-only. This checks the
    conversion is present in the source rather than waiting for a GPU to prove
    it, since without a device `to_host` is a no-op and the omission would not
    show up in any CPU test.
    """
    src = (SRC / "em.py").read_text()
    assert "to_host(" in src, (
        "src/em.py never calls to_host, so the GPU path would leak cupy arrays "
        "into the numpy-only M-step."
    )
