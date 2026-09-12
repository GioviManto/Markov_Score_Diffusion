"""An explicit GPU request must fail rather than fall back to numpy.

SEPARATE FILE ON PURPOSE. tests/test_backend_parity.py carries a module-level
`skipif(not gpu_available())`, which is right for the parity tests -- they need
a device to compare against -- and fatal for these, which exist precisely to
check what happens when there is NO device. Putting them there meant they
skipped on every CPU machine, i.e. exactly where they are the only thing being
tested. That is the same class of hole as the one they were written to close.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.backend import get_xp, gpu_available


def test_explicit_gpu_request_raises_when_no_device_is_present():
    """BP_DEVICE=gpu must fail, not warn and return numpy.

    The old behaviour returned numpy with a RuntimeWarning, on the reasoning
    that slow correct numbers beat none. The cost of that reasoning was a month
    in which EM sweeps labelled GPU ran on the CPU with nothing objecting, so
    an explicit request is now a promise the caller can rely on.
    """
    if gpu_available():
        pytest.skip("a device is present; this test asserts the no-device path")
    with pytest.raises(RuntimeError, match="requested explicitly"):
        get_xp("gpu")


def test_auto_is_the_setting_that_degrades():
    """`auto` keeps the old behaviour, so nothing has lost the ability to."""
    if gpu_available():
        assert get_xp("auto") is not np
        return
    with pytest.warns(RuntimeWarning, match="no usable GPU"):
        assert get_xp("auto") is np
