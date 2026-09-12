"""Shared matplotlib conventions: one place for figure style and saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "figure.figsize": (7.2, 4.6),
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10.5,
    "legend.fontsize": 9,
    "lines.markersize": 4,
}


def new_figure(**kwargs):
    plt.rcParams.update(STYLE)
    return plt.subplots(**kwargs)


def save_figure(fig, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
