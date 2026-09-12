"""CIFAR-10 luminance loading, with the normalisation the OU convention needs.

The forward process of `src/noising.py` is variance preserving *for unit-variance
clean data*: alpha_t^2 + Delta_t = 1 holds as a statement about the noisy
marginal only if Var(a) = 1. So pixels are standardised by statistics computed on
the training split alone -- one global mean and one global standard deviation,
not per-pixel, because a per-pixel standardisation would destroy the spatial
stationarity that the wavelet subbands inherit.

No torchvision: the canonical archive is a set of pickles of uint8 arrays, and
reading them directly costs twenty lines and removes a heavy dependency that is
absent from both this machine and the cluster.
"""

from __future__ import annotations

import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ITU-R BT.601 luma weights -- the standard greyscale reduction, and the one
# whose coefficients are reported alongside CIFAR results elsewhere.
_LUMA = np.array([0.299, 0.587, 0.114])

_TRAIN_BATCHES = tuple(f"data_batch_{i}" for i in range(1, 6))
_TEST_BATCH = "test_batch"


@dataclass(frozen=True)
class ImageSplit:
    """Standardised luminance images plus the statistics used to make them."""

    images: np.ndarray   # (B, 32, 32), float, standardised
    labels: np.ndarray   # (B,) int
    mean: float
    std: float

    def unstandardise(self, x: np.ndarray) -> np.ndarray:
        """Back to the [0, 255] luminance scale (unclipped)."""
        return x * self.std + self.mean


def _read_batches(archive: Path, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    data, labels = [], []
    wanted = {f"cifar-10-batches-py/{n}" for n in names}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name not in wanted:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            payload = pickle.load(handle, encoding="bytes")
            data.append(np.asarray(payload[b"data"], dtype=np.uint8))
            labels.append(np.asarray(payload[b"labels"], dtype=int))
    if not data:
        raise FileNotFoundError(f"none of {sorted(wanted)} found in {archive}")
    return np.concatenate(data), np.concatenate(labels)


def _to_luminance(flat: np.ndarray) -> np.ndarray:
    """(B, 3072) uint8 in R|G|B plane order -> (B, 32, 32) float luminance."""
    rgb = flat.reshape(-1, 3, 32, 32).astype(float)
    return np.tensordot(_LUMA, rgb, axes=([0], [1]))


def load_cifar10_luminance(
    archive: str | Path,
    split: str = "train",
    n_images: int | None = None,
    seed: int = 0,
    stats: tuple[float, float] | None = None,
) -> ImageSplit:
    """Load one split as standardised 32x32 luminance images.

    `stats` forces a given (mean, std); pass the *training* pair when loading the
    test split so the two are on a common scale. Omitting it on the test split
    would leak test statistics into the normalisation.
    """
    archive = Path(archive)
    names = _TRAIN_BATCHES if split == "train" else (_TEST_BATCH,)
    flat, labels = _read_batches(archive, names)
    images = _to_luminance(flat)

    if n_images is not None and n_images < len(images):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(images), size=n_images, replace=False)
        pick.sort()
        images, labels = images[pick], labels[pick]

    if stats is None:
        mean, std = float(images.mean()), float(images.std())
    else:
        mean, std = float(stats[0]), float(stats[1])
    return ImageSplit((images - mean) / std, labels, mean, std)
