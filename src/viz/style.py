from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

CANVAS = "#ffffff"
SURFACE = "#ffffff"
INK = "#000000"
MUTED = "#000000"
GRID = "#d8cbbb"
GOLD = "#c3a46f"
CRIMSON = "#9f3c36"
OCEAN = "#315f72"
FOREST = "#546b4f"

MODALITY_COLORS = {
    "eeg": OCEAN,
    "emg": FOREST,
    "fusion": CRIMSON,
    "neutral": MUTED,
    "accent": GOLD,
}

PHASE_NAMES = ["Preparation", "Momentum", "Extension", "Stabilization"]
PHASE_COLORS = ["#6f7ea0", "#c47b4b", "#6b8a61", "#a58a63"]

SERIF_STACK = [
    "CMU Serif",
    "Computer Modern Roman",
    "Latin Modern Roman",
    "DejaVu Serif",
    "Times New Roman",
    "Times",
]


def resolve_serif_stack() -> list[str]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    stack = [name for name in SERIF_STACK if name in available]
    if "CMU Serif" not in stack:
        warnings.warn("CMU Serif not found. Falling back to the next available serif font.", stacklevel=2)
    if "DejaVu Serif" not in stack:
        stack.append("DejaVu Serif")
    return stack


def apply_scientific_luxe_style(context: str = "figure") -> dict[str, Any]:
    serif_stack = resolve_serif_stack()
    base_font = 10.2 if context == "video" else 9.8
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": serif_stack,
        "font.size": base_font,
        "axes.titlesize": base_font + 1.0,
        "axes.labelsize": base_font - 0.2,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#cfc4b6",
        "axes.grid": False,
        "axes.facecolor": SURFACE,
        "figure.facecolor": CANVAS,
        "savefig.facecolor": CANVAS,
        "savefig.edgecolor": CANVAS,
        "xtick.labelsize": base_font - 1.0,
        "ytick.labelsize": base_font - 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "mathtext.fontset": "cm",
        "figure.dpi": 220,
        "savefig.dpi": 300,
    })
    return {"font_stack": serif_stack, "font_primary": serif_stack[0]}


def style_axes(ax: plt.Axes, *, grid_axis: str | None = None, hide_ticks: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.35)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def style_colorbar(cbar: Any, label: str | None = None) -> None:
    if label:
        cbar.set_label(label, color=INK)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(colors=INK, labelsize=8)


def add_phase_bands(ax: plt.Axes, edges: list[float], *, alpha: float = 0.08) -> None:
    left = 0.0
    full_edges = [*edges, None]
    for idx, right in enumerate(full_edges):
        if right is None:
            ax.axvspan(left, ax.get_xlim()[1], color=PHASE_COLORS[idx], alpha=alpha, linewidth=0)
            break
        ax.axvspan(left, right, color=PHASE_COLORS[idx], alpha=alpha, linewidth=0)
        ax.axvline(right, color="white", linestyle="--", linewidth=0.8, alpha=0.75)
        left = right


def add_corner_label(ax: plt.Axes, text: str) -> None:
    kwargs = {
        "ha": "left",
        "va": "top",
        "fontsize": 8.5,
        "color": MUTED,
    }
    if hasattr(ax, "text2D"):
        ax.text2D(0.02, 0.98, text, transform=ax.transAxes, **kwargs)
        return
    ax.text(0.02, 0.98, text, transform=ax.transAxes, **kwargs)


def save_figure(fig: plt.Figure, png_path: Path) -> None:
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=CANVAS)
    plt.close(fig)


def style_3d_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    try:
        ax.xaxis.pane.set_facecolor(mpl.colors.to_rgba(SURFACE, 1.0))
        ax.yaxis.pane.set_facecolor(mpl.colors.to_rgba(SURFACE, 1.0))
        ax.zaxis.pane.set_facecolor(mpl.colors.to_rgba(SURFACE, 1.0))
        ax.xaxis.pane.set_edgecolor(GRID)
        ax.yaxis.pane.set_edgecolor(GRID)
        ax.zaxis.pane.set_edgecolor(GRID)
    except Exception:
        pass


def short_phase_name(phase_idx: int) -> str:
    return ["Prep", "Momentum", "Extend", "Stable"][phase_idx]


def abbreviate_muscles(muscles: list[str]) -> list[str]:
    mapping = {
        "quad": "QD",
        "ham": "HM",
        "glute": "GL",
        "ta": "TA",
        "gast": "GS",
        "erector": "ER",
    }
    return [mapping.get(m, m[:3].upper()) for m in muscles]


def quantile_limits(values: np.ndarray, lower: float = 0.02, upper: float = 0.98) -> tuple[float, float]:
    lo, hi = np.quantile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(values))
        hi = float(np.max(values))
    return float(lo), float(hi)
