import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config):
    """Register the `slow` marker.

    Used by the wavelet-model tests, which fit a model end to end and take
    minutes rather than seconds. Registering it keeps `-m "not slow"` available
    and stops pytest warning that the marker is a typo.
    """
    config.addinivalue_line(
        "markers", "slow: end-to-end fits that take minutes, not seconds"
    )
