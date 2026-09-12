"""Figure conventions for the thesis.

One place for palette, sizing, and saving, so that every figure in the document
looks like it came from the same hand. Three rules the old figures broke:

  * legends never sit inside the data area (they collided with tick labels in
    the three-panel EM diagnostics and with the curves in the non-Markov panel);
  * panels are laid out by ``constrained`` and never by ``tight_layout``, which
    is what let the third panel's y-axis labels be overdrawn;
  * widths are fixed to the thesis text block, so no figure is ever rescaled by
    LaTeX and the type size is the same in all of them.

Nothing here writes to any shared figures folder. The thesis carries its own
copies in ``thesis/figures`` so that restyling it cannot disturb the
paper, the workshop note, or the compendium.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- geometry --
# A4 with 2.5 cm side margins leaves a 160 mm text block.
FULL = 6.30   # inches, the full text width
HALF = 3.05   # two side by side with a gap
WIDE = 6.30

# ------------------------------------------------------------------ colour --
# Okabe-Ito, which the project already used in part. Safe for the common forms
# of colour blindness and separable in greyscale, which matters because this
# document will be printed.
BLUE   = "#0072B2"
VERM   = "#D55E00"
GREEN  = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY    = "#56B4E9"
GREY   = "#4D4D4D"
CYCLE = [BLUE, VERM, GREEN, PURPLE, ORANGE, SKY]

# For anything ordered by diffusion time, a ramp rather than the cycle: the
# reader should see the ordering without consulting the legend.
# hi stops short of viridis' yellow end: at 0.90 the last series was too
# light to read on paper, which is where this document is examined.
def ramp(n: int, cmap: str = "viridis", lo: float = 0.06, hi: float = 0.74):
    c = matplotlib.colormaps[cmap]
    if n == 1:
        return [c(0.5)]
    return [c(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Body text is 12 pt; captions render at 10 pt. Figure text sits just below
    # the caption so it never looks larger than the prose around it.
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    # Sans-serif throughout: cleaner at small sizes, and the text reads
    # unimpeded against the data. Serif fonts (Times especially) in a
    # graphic are the opposite of what a figure is for.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Mathtext in Arial as well, not the DejaVu math faces: mixing the two
    # families inside one label is what made the earlier figures read as
    # part-serif. Symbols Arial lacks fall back to STIX sans, still sans.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.cal": "Arial:italic",
    "mathtext.fallback": "stixsans",
    # Embed text as TrueType (Type 42), not the matplotlib default Type 3:
    # Type 3 glyphs render poorly in some viewers and print pipelines, and
    # the reproducibility appendix used to have to disclose them.
    "pdf.fonttype": 42,
    "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.titlelocation": "left",
    "axes.titlepad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.0,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
}


def new_figure(nrows: int = 1, ncols: int = 1, width: float = FULL,
               height: float = 2.6, **kw):
    """A figure sized to the text block, with the house style applied."""
    plt.rcParams.update(STYLE)
    return plt.subplots(nrows, ncols, figsize=(width, height), **kw)


def legend_above(ax, ncol: int = 3, **kw):
    """Put the legend in its own band above the axes.

    The default matplotlib placement searches for empty space *inside* the
    axes; when there is none it overlaps the data or the tick labels, which is
    what happened in three of the eighteen figures this replaces.
    """
    return ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.12),
                     mode="expand", ncol=ncol, borderaxespad=0.0,
                     frameon=False, **kw)


def label_lines(ax, entries, dx: float = 1.02, fontsize: float = 8.5):
    """Label curves at their right-hand end instead of using a legend.

    ``entries`` is a sequence of ``(text, y, colour)``. Reading a curve should
    not require a lookup when there are only a few of them.
    """
    for text, y, colour in entries:
        ax.annotate(text, xy=(dx, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=fontsize, color=colour)


def save(fig, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {p.with_suffix('.pdf').name}")
