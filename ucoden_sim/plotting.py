"""Shared plotting conventions for Figs. 3-5."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated colour-blind-safe categorical order (fixed slots; never cycle beyond what is used)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#52514e"
GRID = "#e6e5e1"


def style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, path_no_ext: str):
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png", dpi=200)
