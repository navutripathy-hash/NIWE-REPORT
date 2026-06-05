"""
goal_w2_offshore.py
Goal W-2: Offshore Wind Capacity — Target: 30 GW by 2030
Model: Scenario Analysis (no historical installation data — all zeros)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import apply_style, add_status_badge, save_and_close, COLORS

TARGET_GW   = 30.0
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

# ── Scenario Definitions ──────────────────────────────────────────────────────
# Scenario keys: year → cumulative GW
SCENARIOS = {
    "Worst Case\n(No project launched)": {
        "years": [2015,2018,2020,2022,2024,2026,2027,2028,2029,2030],
        "gw":    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        "color": COLORS["scenario_pes"],
        "ls":    "-",
        "status":"NOT ACHIEVED",
        "fc2030": 0,
    },
    "Base Case\n(1 pilot by 2028)": {
        "years": [2015,2018,2020,2022,2024,2026,2027,2028,2029,2030],
        "gw":    [0,   0,   0,   0,   0,   0,   0,   0.5, 1.0, 1.5],
        "color": COLORS["scenario_base"],
        "ls":    "-",
        "status":"NOT ACHIEVED",
        "fc2030": 1.5,
    },
    "Optimistic\n(2-3 GW by 2030)": {
        "years": [2015,2018,2020,2022,2024,2026,2027,2028,2029,2030],
        "gw":    [0,   0,   0,   0,   0,   0,   0.2, 0.8, 1.8, 3.0],
        "color": COLORS["scenario_opt"],
        "ls":    "-.",
        "status":"NOT ACHIEVED",
        "fc2030": 3.0,
    },
    "Aggressive\n(VGF surge + intl devs)": {
        "years": [2015,2018,2020,2022,2024,2026,2027,2028,2029,2030],
        "gw":    [0,   0,   0,   0,   0,   0,   0.5, 2.0, 4.5, 8.0],
        "color": "#2c3e50",
        "ls":    "--",
        "status":"NOT ACHIEVED",
        "fc2030": 8.0,
    },
}

# Policy timeline milestones
MILESTONES = [
    (2015, "Offshore Wind\nPolicy Notified"),
    (2018, "NIWE EoI\nIssued"),
    (2023, "Offshore Strategy\nReleased"),
    (2024, "VGF ₹74.5 bn\nApproved (1 GW)"),
    (2026, "New TN Tender\nExpected"),
]


def run(plots_dir=PLOTS_DIR):
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#fafafa")

    # ── Panel 1: Scenario Trajectories ────────────────────────────────────────
    ax1 = axes[0]
    for name, sc in SCENARIOS.items():
        ax1.plot(sc["years"], sc["gw"], color=sc["color"],
                 lw=2.2, ls=sc["ls"], label=f"{name.replace(chr(10),' ')}  →2030: {sc['fc2030']} GW",
                 marker="o", ms=4, zorder=4)

    # Target line
    ax1.axhline(y=TARGET_GW, color=COLORS["target"], lw=2, linestyle="--",
                label=f"Target: {TARGET_GW} GW")

    # Policy milestones
    for yr, label in MILESTONES:
        ax1.axvline(yr, color="#bdc3c7", lw=0.8, linestyle=":")
        ax1.text(yr, 31.5, label, fontsize=6.5, ha="center", va="bottom",
                 color="#7f8c8d", rotation=0)

    # Highlight zero history
    ax1.axvspan(2015, 2026, alpha=0.06, color="#e74c3c",
                label="Zero Offshore Installed (2015–2026)")
    ax1.text(2020.5, 14, "ZERO OFFSHORE\nINSTALLED\n(2015–2026)", fontsize=10,
             ha="center", va="center", color="#c0392b", fontweight="bold",
             alpha=0.5)

    ax1.set_title("GOAL W-2 · Offshore Wind — Scenario Trajectories\n(Historical: 0 MW | Target: 30 GW by 2030)")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Cumulative Offshore Capacity (GW)")
    ax1.legend(loc="upper left", fontsize=7.5)
    ax1.set_xlim(2014, 2031); ax1.set_ylim(-1, 35)
    add_status_badge(ax1, "NOT ACHIEVED")

    # ── Panel 2: 2030 Forecast Bar + Gap ──────────────────────────────────────
    ax2 = axes[1]
    names   = [n.replace("\n"," ") for n in SCENARIOS.keys()]
    fc_vals = [sc["fc2030"] for sc in SCENARIOS.values()]
    bar_clrs = [sc["color"] for sc in SCENARIOS.values()]

    bars = ax2.bar(range(len(names)), fc_vals, color=bar_clrs, width=0.55,
                   edgecolor="white", linewidth=1.5, zorder=3)
    for bar, val in zip(bars, fc_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{val} GW", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    ax2.axhline(y=TARGET_GW, color=COLORS["target"], lw=2.2, linestyle="--",
                label=f"Target: {TARGET_GW} GW")
    ax2.fill_between([-0.5, 3.5], [0]*2, [TARGET_GW]*2,
                     color="#fadbd8", alpha=0.25, label="Gap zone")

    # Gap arrows
    for i, val in enumerate(fc_vals):
        gap = TARGET_GW - val
        if gap > 0:
            ax2.annotate("", xy=(i, TARGET_GW), xytext=(i, val + 0.1),
                         arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.5))
            ax2.text(i + 0.27, (val + TARGET_GW)/2,
                     f"Gap\n−{gap:.1f} GW", fontsize=7.5, color="#c0392b",
                     fontweight="bold", va="center")

    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(["Worst\nCase","Base\nCase","Optimistic","Aggressive"],
                        fontsize=9)
    ax2.set_title("GOAL W-2 · 2030 Forecast by Scenario\n(All scenarios miss 30 GW target by wide margin)")
    ax2.set_ylabel("Forecast 2030 Offshore Capacity (GW)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 35)
    add_status_badge(ax2, "NOT ACHIEVED")

    fig.suptitle("GOAL W-2 · India Offshore Wind Capacity · Target: 30 GW by 2030\n"
                 "Current Status: 0 MW Installed | Physically Impossible to Achieve 30 GW by 2030",
                 fontsize=13, fontweight="bold", color="#1a3a5c", y=1.03)

    return save_and_close(fig, "W2_offshore_wind", plots_dir)


if __name__ == "__main__":
    run()
