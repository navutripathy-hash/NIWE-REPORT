"""
goal_n2_nonfossil_share.py
Goal N-2: Non-Fossil Share of Installed Capacity — Target: 50% by 2030
STATUS: ALREADY ACHIEVED in June 2025 — 5 years ahead of schedule.
Model: Historical trend + milestone achievement visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils import (apply_style, add_target_line, add_status_badge,
                   save_and_close, COLORS)

TARGET_PCT  = 50.0
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

YEARS       = np.array([2015,2016,2017,2018,2019,2020,2021,2022,
                         2023,2024,2025,2026], dtype=float)
NONFOSSIL_SHARE = np.array([28.0, 29.5, 30.5, 32.0, 33.5, 36.0, 38.5, 41.0,
                              43.0, 45.5, 50.0, 52.0])  # % of total installed

TOTAL_CAP   = np.array([300, 315, 330, 345, 360, 375, 390, 410,
                          450, 495, 550, 600], dtype=float)  # GW approx
NONFOSSIL_GW = NONFOSSIL_SHARE / 100 * TOTAL_CAP


def run(plots_dir=PLOTS_DIR):
    apply_style()

    X  = YEARS.reshape(-1, 1)
    m  = LinearRegression().fit(X, NONFOSSIL_SHARE)
    r2 = r2_score(NONFOSSIL_SHARE, m.predict(X))
    pred_2030 = float(m.predict([[2030]])[0])
    all_yrs = np.arange(2015, 2032)
    pred_all = m.predict(all_yrs.reshape(-1, 1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor("#fafafa")

    # ── Panel 1: Share Trend with Achievement Marker ──────────────────────────
    ax1 = axes[0]
    ax1.plot(YEARS, NONFOSSIL_SHARE, color=COLORS["historical"], lw=2.5,
             marker="o", ms=6, zorder=5, label="Actual Non-Fossil Share (%)")
    ax1.plot(all_yrs, pred_all, color=COLORS["linear"], lw=1.8, ls="--",
             alpha=0.7, label=f"Linear Reg (R²={r2:.4f})  →2030: {pred_2030:.1f}%")

    # Highlight achievement point (2025)
    ax1.scatter([2025], [50.0], s=200, color=COLORS["achieved"], zorder=8,
                edgecolors="white", linewidth=2, label="✓ Target ACHIEVED (Jun 2025)")
    ax1.annotate("50% NDC Target\nACHIEVED ✓\n(June 2025 — 5 yrs early!)",
                 xy=(2025, 50), xytext=(2021.5, 53.5),
                 fontsize=8.5, color=COLORS["achieved"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COLORS["achieved"], lw=1.5))

    ax1.axvline(2026.5, color="#aaa", lw=1, ls=":")
    add_target_line(ax1, TARGET_PCT, "NDC Target: 50%", color="#c0392b")
    ax1.fill_between(YEARS, NONFOSSIL_SHARE, TARGET_PCT,
                     where=(NONFOSSIL_SHARE >= TARGET_PCT),
                     color=COLORS["achieved"], alpha=0.12, label="Above target (achieved)")

    for yr, val in zip(YEARS, NONFOSSIL_SHARE):
        if yr >= 2020:
            ax1.annotate(f"{val:.0f}%", (yr, val), textcoords="offset points",
                         xytext=(3, 6), fontsize=7.5, color=COLORS["historical"])

    add_status_badge(ax1, "ACHIEVED")
    ax1.set_title("GOAL N-2 · Non-Fossil Share of Total Installed Capacity\n(ACHIEVED 5 Years Ahead of 2030 Target)")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Non-Fossil Share (%)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_ylim(20, 65); ax1.set_xlim(2014, 2031)

    # ── Panel 2: Speed of Achievement Timeline ────────────────────────────────
    ax2 = axes[1]
    milestones_yr  = [2015, 2020, 2022, 2023, 2024, 2025, 2026]
    milestones_pct = [28, 36, 41, 43, 45.5, 50, 52]
    milestone_labels = [
        "28%\n(2015)",
        "36%\n(2020)",
        "41%\n(2022)",
        "43%\n(2023)",
        "45.5%\n(2024)",
        "50% ✓\n(2025)",
        "52%\n(2026)",
    ]

    colors_ms = [COLORS["historical"]] * 5 + [COLORS["achieved"]] + [COLORS["achieved"]]
    sizes_ms  = [60]*5 + [200] + [160]

    ax2.plot(milestones_yr, milestones_pct, color=COLORS["historical"], lw=2)
    for yr, pct, lbl, c, s in zip(milestones_yr, milestones_pct,
                                    milestone_labels, colors_ms, sizes_ms):
        ax2.scatter(yr, pct, s=s, color=c, zorder=6, edgecolors="white", lw=1.5)
        ax2.annotate(lbl, (yr, pct), textcoords="offset points",
                     xytext=(-5, 10), fontsize=8, ha="center",
                     color=c, fontweight="bold" if pct >= 50 else "normal")

    ax2.axhline(TARGET_PCT, color="#c0392b", lw=2.2, ls="--", label="Target: 50%")
    ax2.fill_between(milestones_yr, milestones_pct, 0, alpha=0.1,
                     color=COLORS["historical"])
    ax2.fill_between([2025, 2031], [50, 53], [50]*2, alpha=0.2,
                     color=COLORS["achieved"])

    ax2.text(2021, 22, "India crossed 50% non-fossil share\n5 years ahead of the NDC 2030 deadline.",
             fontsize=10, style="italic", color=COLORS["achieved"],
             ha="center", va="center",
             bbox=dict(facecolor="white", alpha=0.7, edgecolor=COLORS["achieved"],
                       boxstyle="round,pad=0.4"))

    add_status_badge(ax2, "ACHIEVED")
    ax2.set_title("GOAL N-2 · Milestone Timeline\n(50% NDC Target — Already Crossed)")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Non-Fossil Share (%)")
    ax2.legend(fontsize=9)
    ax2.set_ylim(15, 65); ax2.set_xlim(2014, 2027)

    fig.suptitle("GOAL N-2 · Non-Fossil Share ≥ 50% by 2030 · STATUS: ✓ ACHIEVED (June 2025)",
                 fontsize=14, fontweight="bold", color=COLORS["achieved"], y=1.02)

    return save_and_close(fig, "N2_nonfossil_share", plots_dir)


if __name__ == "__main__":
    run()
