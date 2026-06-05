"""
goal_dashboard.py
Master Dashboard: All Goals at a Glance — Forecast Summary for 2030
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import apply_style, save_and_close, COLORS, STATUS_COLORS

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

GOALS = [
    # (id, name, target, forecast, unit, status, source_model)
    ("W-1", "Total Wind\nCapacity",   100,   90,     "GW",     "AT RISK",         "Poly+ARIMA+XGB"),
    ("W-2", "Offshore\nWind",          30,    1.5,   "GW",     "NOT ACHIEVED",    "Scenario"),
    ("W-3", "Manufacturing\nLocal%",   85,   82,     "%",      "LIKELY ACHIEVED", "Linear Reg"),
    ("I-1", "ISTS\nTransmission",   50890, 35000,    "ckt km", "AT RISK",         "Linear Reg"),
    ("N-1", "Non-Fossil\nCapacity",   500,  420,     "GW",     "AT RISK",         "Poly Reg"),
    ("N-2", "Non-Fossil\nShare",       50,   52,     "%",      "ACHIEVED",        "Achieved 2025"),
    ("N-3", "Wind\nRPO/RCO",         6.94,  6.5,    "%",      "LIKELY ACHIEVED", "Gen Forecast"),
]

def run(plots_dir=PLOTS_DIR):
    apply_style()

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#f0f3f7")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.6)

    # ── Panel 1: Normalized Progress Bar Chart ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    n = len(GOALS)
    y_pos = np.arange(n)

    for i, (gid, name, target, forecast, unit, status, model) in enumerate(GOALS):
        pct   = min(forecast / target * 100, 115)
        color = STATUS_COLORS.get(status, "#7f8c8d")
        # Background bar (target = 100%)
        ax1.barh(i, 115, left=0, color="#e8ecf0", height=0.65, zorder=1)
        # Actual bar
        ax1.barh(i, pct, left=0, color=color, height=0.65, alpha=0.85, zorder=2)
        # Target line at 100%
        ax1.axvline(100, color="#2c3e50", lw=1.5, ls="--", alpha=0.6, zorder=3)
        # Labels
        ax1.text(pct + 1.5, i, f"{forecast} {unit}", va="center", fontsize=9,
                 fontweight="bold", color=color)
        ax1.text(-1, i, f"{gid}: {name.replace(chr(10),' ')}", va="center",
                 ha="right", fontsize=9, color="#2c3e50")
        ax1.text(102, i, f"Target: {target} {unit}", va="center", fontsize=8,
                 color="#7f8c8d")

    ax1.set_xlim(-25, 130)
    ax1.set_yticks([])
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xticklabels(["0%", "25%", "50%", "75%", "100% = Target"])
    ax1.set_title("Progress Toward 2030 Target (% of Target Achieved by Forecast)",
                  fontsize=13, fontweight="bold")
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.grid(False)

    # Legend
    legend_items = [mpatches.Patch(color=c, label=s)
                    for s, c in STATUS_COLORS.items()]
    ax1.legend(handles=legend_items, loc="lower right", fontsize=9,
               title="Goal Status", title_fontsize=9)

    # ── Panel 2: Absolute Forecast vs Target ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    # Normalise all to a 0-1 scale for comparison
    norms = [(forecast/target) for _, _, target, forecast, _, _, _ in GOALS]
    bar_colors = [STATUS_COLORS.get(status, "#7f8c8d")
                  for _, _, _, _, _, status, _ in GOALS]
    labels = [f"{gid}\n{name.replace(chr(10),' ')}" for gid, name, *_ in GOALS]

    bars = ax2.bar(range(n), norms, color=bar_colors, width=0.55,
                   edgecolor="white", lw=1.5, zorder=3)
    ax2.axhline(1.0, color="#2c3e50", lw=2, ls="--", label="Target = 1.0", zorder=4)
    ax2.fill_between([-0.5, n-0.5], [1.0]*2, [1.3]*2,
                     color="#d5f5e3", alpha=0.3, label="Above target")
    ax2.fill_between([-0.5, n-0.5], [0]*2, [1.0]*2,
                     color="#fadbd8", alpha=0.18, label="Below target")

    for bar, (gid, name, target, forecast, unit, status, model) in zip(bars, GOALS):
        pct_lbl = f"{forecast/target*100:.0f}%\n({forecast} {unit})"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 pct_lbl, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Forecast / Target Ratio")
    ax2.set_ylim(0, 1.35)
    ax2.legend(fontsize=9)
    ax2.set_title("Forecast-to-Target Ratio for Each Goal  (1.0 = exactly on target)",
                  fontsize=13, fontweight="bold")

    # ── Panel 3: Summary Table ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    col_headers = ["Goal ID", "Goal Name", "Target", "Forecast 2030",
                   "Gap", "Unit", "Status", "Best Model"]
    rows = []
    for gid, name, target, forecast, unit, status, model in GOALS:
        gap = target - forecast
        gap_str = f"−{abs(gap):.1f}" if gap > 0 else f"+{abs(gap):.1f}"
        rows.append([gid, name.replace("\n"," "), str(target), str(forecast),
                     gap_str, unit, status, model])

    table = ax3.table(cellText=rows, colLabels=col_headers,
                      cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for j in range(len(col_headers)):
        table[0, j].set_facecolor("#1a3a5c")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows by status
    status_col = 6
    for i, (_, _, _, _, _, status, _) in enumerate(GOALS, start=1):
        color = STATUS_COLORS.get(status, "#7f8c8d")
        for j in range(len(col_headers)):
            if j == status_col:
                table[i, j].set_facecolor(color)
                table[i, j].set_text_props(color="white", fontweight="bold")
            else:
                table[i, j].set_facecolor("#f7f9fb" if i % 2 == 0 else "white")

    ax3.set_title("Summary Table: All Goals — Forecast, Gap & Status",
                  fontsize=13, fontweight="bold", pad=20)

    fig.suptitle("INDIA WIND ENERGY 2030 — MASTER DASHBOARD\nAll Goals: Forecast, Gap Analysis & Status",
                 fontsize=16, fontweight="bold", color="#1a3a5c", y=0.99)

    return save_and_close(fig, "00_master_dashboard", plots_dir)


if __name__ == "__main__":
    run()
