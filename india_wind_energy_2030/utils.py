"""
utils.py
Shared plotting utilities and styling for India Wind Energy 2030 Project.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "historical":   "#1a3a5c",   # deep navy
    "poly2":        "#e07b39",   # burnt orange
    "poly3":        "#2e8b57",   # sea green
    "arima":        "#c0392b",   # crimson
    "xgboost":      "#8e44ad",   # purple
    "linear":       "#2980b9",   # blue
    "scenario_pes": "#e74c3c",   # red
    "scenario_base":"#f39c12",   # amber
    "scenario_opt": "#27ae60",   # green
    "target":       "#2c3e50",   # dark charcoal
    "achieved":     "#16a085",   # teal
    "gap_fill":     "#fadbd8",   # light pink
    "forecast_fill":"#d6eaf8",   # light blue
}

GOAL_TARGETS = {
    "W1": {"value": 100, "unit": "GW",      "label": "Target: 100 GW"},
    "W2": {"value": 30,  "unit": "GW",      "label": "Target: 30 GW"},
    "W3": {"value": 85,  "unit": "%",       "label": "Target: 85%"},
    "I1": {"value": 50890,"unit":"ckt km",  "label": "Target: 50,890 ckt km"},
    "N1": {"value": 500, "unit": "GW",      "label": "Target: 500 GW"},
    "N2": {"value": 50,  "unit": "%",       "label": "Target: 50%"},
    "N3": {"value": 6.94,"unit": "%",       "label": "Target: 6.94%"},
}

STATUS_COLORS = {
    "ACHIEVED":        "#16a085",
    "LIKELY ACHIEVED": "#27ae60",
    "AT RISK":         "#e67e22",
    "NOT ACHIEVED":    "#c0392b",
}

def apply_style():
    """Apply consistent global style to all plots."""
    plt.rcParams.update({
        "figure.facecolor": "#fafafa",
        "axes.facecolor":   "#fafafa",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "DejaVu Sans",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "legend.framealpha":0.85,
    })


def add_target_line(ax, target_val, label, color=None):
    """Draw a horizontal dashed target line."""
    c = color or COLORS["target"]
    ax.axhline(y=target_val, color=c, linestyle="--", linewidth=1.8,
               alpha=0.85, label=label, zorder=4)


def add_forecast_band(ax, x, lower, upper, color=None, alpha=0.18):
    """Shade a confidence / scenario band."""
    c = color or COLORS["forecast_fill"]
    ax.fill_between(x, lower, upper, color=c, alpha=alpha, zorder=1)


def add_status_badge(ax, status: str, x=0.97, y=0.94):
    """Place a coloured status badge in the top-right corner of an axes."""
    color = STATUS_COLORS.get(status, "#7f8c8d")
    ax.text(x, y, status, transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="white",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color, alpha=0.90,
                      edgecolor="none"))


def annotate_gap(ax, year, forecast_val, target_val, unit="GW"):
    """Draw a double-headed arrow annotating the gap between forecast and target."""
    if abs(forecast_val - target_val) < 0.01:
        return
    mid = (forecast_val + target_val) / 2
    gap  = abs(target_val - forecast_val)
    ax.annotate("", xy=(year, target_val), xytext=(year, forecast_val),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.5))
    ax.text(year + 0.15, mid,
            f"Gap\n{gap:.1f} {unit}", fontsize=8, color="#c0392b",
            va="center", fontweight="bold")


def save_and_close(fig, goal_id: str, plots_dir: str):
    """Save figure as PNG and close it."""
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"goal_{goal_id.lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Saved: {path}")
    return path
