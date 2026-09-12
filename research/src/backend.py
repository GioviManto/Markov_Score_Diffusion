"""Array-module dispatch, so the BP recursion can run on CPU or GPU from one code path.

Why one code path rather than two
---------------------------------
Everything in this project rests on grid BP being *exact*: it agrees with the closed-form
Gaussian score to 9.2e-15 and with brute-force enumeration of all M^n configurations to
1.0e-14. A second, separately written GPU implementation would put that guarantee at risk in
the least visible way possible -- the two would agree on the easy cases and drift somewhere
nobody looks.

So there is no GPU implementation. There is one implementation, parameterised by its array
module, and `tests/test_backend_parity.py` asserts the two devices return bitwise-comparable
results on the same inputs. If they ever diverge, that test fails rather than a paper number
quietly changing.

Precision
---------
Float64 throughout, deliberately. The H100/H200 cards here do fp64 at a few tens of TFLOP/s,
which is ample, and the alternative is not worth discussing: at fp32 the message
normalisation would lose roughly eight significant digits, which is most of the margin the
exactness claims live in.

Usage
-----
    from src.backend import get_xp, to_device, to_host

    xp = get_xp()                     # numpy, or cupy if BP_DEVICE=gpu
    K = to_device(K_host, xp)
    means, _ = grid_bp_batch(..., xp=xp)
    means = to_host(means)

Selection is by the ``BP_DEVICE`` environment variable rather than a function argument
threaded through every call site, so a batch script chooses the device once and nothing else
in the package needs to know.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

_CUPY = None
_CUPY_CHECKED = False


def _try_import_cupy():
    global _CUPY, _CUPY_CHECKED
    if not _CUPY_CHECKED:
        _CUPY_CHECKED = True
        try:
            import cupy  # noqa: F401
            _CUPY = cupy
        except Exception:  # pragma: no cover - depends on the machine
            _CUPY = None
    return _CUPY


def gpu_available() -> bool:
    """True only if cupy imports *and* a device actually responds.

    The import succeeding is not enough: cupy installs fine on a login node with no GPU, and
    the failure would otherwise surface deep inside a BP sweep rather than at startup.
    """
    cp = _try_import_cupy()
    if cp is None:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        # A matmul, not just an allocation. Allocation succeeds without cuBLAS loaded, so an
        # allocation-only probe reports a usable GPU on a node where every matrix product
        # will fail -- which is exactly what happened on the first cluster attempt
        # (libcublas.so.12 missing because the cuda module was not loaded in the batch job).
        # The probe has to exercise the operation the recursion actually depends on.
        a = cp.ones((8, 8), dtype=cp.float64)
        float((a @ a).sum())
        return True
    except Exception:  # pragma: no cover - depends on the machine
        return False


def get_xp(device: str | None = None) -> Any:
    """Return the array module to compute with.

    ``device`` defaults to the ``BP_DEVICE`` environment variable, then to CPU.

    AN EXPLICIT GPU REQUEST FAILS CLOSED. ``BP_DEVICE=gpu`` used to warn and fall
    back to numpy, on the reasoning that a job landing on a node without a
    visible device should produce correct numbers slowly rather than none. That
    reasoning is wrong for this project, and the history says so: for a month
    ``em.py`` did not import the backend at all, so "GPU" EM sweeps were
    CPU-bound and nothing said so, and a separate GPU gate passed while every
    parity test skipped. A warning that only appears in a log nobody reads is
    indistinguishable from no warning, and the failure it hides -- a timing or a
    device claim attributed to hardware that never ran -- is exactly the kind
    this package cannot afford.

    So the contract is now: ``gpu`` means GPU or an exception. ``auto`` is the
    setting that prefers a device and degrades quietly, and a caller that
    genuinely wants "fast if possible" should ask for that instead.
    """
    if device is None:
        device = os.environ.get("BP_DEVICE", "cpu")
    device = device.lower()

    if device in ("cpu", "numpy"):
        return np
    if device in ("gpu", "cuda", "cupy"):
        if gpu_available():
            return _try_import_cupy()
        raise RuntimeError(
            f"device {device!r} was requested explicitly and no usable GPU is "
            "present. Refusing to fall back to numpy: a run that reports GPU "
            "timings or GPU-device provenance while executing on the CPU is "
            "worse than a run that fails. Use BP_DEVICE=auto to prefer a GPU "
            "and degrade to CPU, or BP_DEVICE=cpu to ask for the CPU."
        )
    if device == "auto":
        if gpu_available():
            return _try_import_cupy()
        warnings.warn(
            "BP_DEVICE=auto found no usable GPU; using numpy. Timings from "
            "this run are CPU timings.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np
    raise ValueError(
        f"unknown device {device!r}; expected 'cpu', 'gpu' or 'auto'"
    )


def to_device(a: np.ndarray, xp: Any) -> Any:
    """Move a host array to ``xp``, preserving dtype (float64 stays float64)."""
    if xp is np:
        return np.asarray(a)
    return xp.asarray(a)


def to_host(a: Any) -> np.ndarray:
    """Bring an array back to numpy, whichever module produced it."""
    cp = _try_import_cupy()
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def device_name(xp: Any) -> str:
    """Short label for provenance, so an output file records where it was computed."""
    if xp is np:
        return "cpu:numpy"
    cp = _try_import_cupy()
    try:  # pragma: no cover - depends on the machine
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        return f"gpu:{props['name'].decode()}"
    except Exception:  # pragma: no cover
        return "gpu:unknown"
