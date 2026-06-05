"""
goal_i1_transmission.py
Goal I-1: ISTS Transmission Lines — Target: 50,890 ckt km by 2030
Models: Linear Regression | Phased Plan vs Likely Achievement
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils import (apply_style, add_target_line, add_forecast_band,
                   add_status_badge, annotate_gap, save_and_close, COLORS)

TARGET_KM   = 50890
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

# Historical actual cumulative ckt km (GEC commissioned + ISTS)
HIST_YEARS  = np.array([2015, 2018, 2020, 2022, 2024, 2025, 2026], dtype=float)
ACTUAL_KM   = np.array([0, 1500, 3200, 3400, 6000, 9000, 13000], dtype=float)

# CEA planned cumulative
PLAN_YEARS  = np.array([2015, 2020, 2022, 2024, 2026, 2027, 2028, 2029, 2030], dtype=float)
PLANNED_KM  = np.array([0, 3200, 5000, 8000, 15000, 25000, 38000, 47000, 50890], dtype=float)

FORECAST_YRS = np.arange(2027, 2031)
ALL_YRS      = np.concatenate([HIST_YEARS, FORECAST_YRS])


def fit_linear(years, values):
    X = years.reshape(-1, 1)
    m = LinearRegression().fit(X, values)
    r2 = r2_score(values, m.predict(X))
    return m, r2


def run(plots_dir=PLOTS_DIR):
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("#fafafa")

    # ── Linear regression on actual commissioned km ──────────────────────────
    model, r2 = fit_linear(HIST_YEARS, ACTUAL_KM)
    lin_pred  = model.predict(ALL_YRS.reshape(-1, 1))
    fc_2030   = float(model.predict([[2030]])[0])
    # Acceleration model (recent years have higher slope)
    recent_yrs  = HIST_YEARS[-3:]
    recent_km   = ACTUAL_KM[-3:]
    acc_model, r2_acc = fit_linear(recent_yrs, recent_km)
    acc_pred    = acc_model.predict(np.arange(2025, 2031).reshape(-1,1))
    acc_fc_2030 = float(acc_model.predict([[2030]])[0])

    # ── Panel 1: Actual vs Planned vs Forecast ─────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(PLAN_YEARS, PLANNED_KM, color=COLORS["forecast_fill"],
                     alpha=0.4, label="CEA Planned Trajectory")
    ax1.plot(PLAN_YEARS, PLANNED_KM, color="#2980b9", lw=1.8, ls="--",
             label="CEA Plan", marker="s", ms=5)
    ax1.plot(HIST_YEARS, ACTUAL_KM, color=COLORS["historical"], lw=2.5,
             marker="o", ms=7, zorder=5, label="Actual Commissioned (ckt km)")
    ax1.plot(ALL_YRS, lin_pred, color=COLORS["poly2"], lw=2,
             label=f"Linear Reg (R²={r2:.3f})  →2030: {fc_2030:,.0f} km", ls="-")
    ax1.plot(np.arange(2025, 2031), acc_pred, color=COLORS["poly3"], lw=2,
             label=f"Accel. Reg (recent 3yr)  →2030: {acc_fc_2030:,.0f} km", ls="-.")
    ax1.axvline(2026.5, color="#aaa", lw=1, ls=":")
    ax1.text(2026.7, 2000, "Forecast →", fontsize=8, color="#777")
    add_target_line(ax1, TARGET_KM, f"Target: {TARGET_KM:,} ckt km", color="#c0392b")
    add_forecast_band(ax1, FORECAST_YRS,
                      [fc_2030 * 0.9] * 4, [acc_fc_2030 * 1.1] * 4)
    add_status_badge(ax1, "AT RISK")
    ax1.set_title("GOAL I-1 · ISTS Transmission\nActual vs Planned vs Forecast")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Cumulative ISTS Lines (ckt km)")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.set_ylim(0, 60000); ax1.set_xlim(2014, 2031)
    # Data labels on actual
    for yr, km in zip(HIST_YEARS, ACTUAL_KM):
        ax1.annotate(f"{km:,}", (yr, km), textcoords="offset points",
                     xytext=(4, 6), fontsize=7.5, color=COLORS["historical"])

    # ── Panel 2: 2030 Forecast vs Target Bar ──────────────────────────────────
    ax2 = axes[1]
    scenarios = {
        "CEA Plan\n(Ideal)":        50890,
        "Linear Reg\n(Trend)":      min(fc_2030, 55000),
        "Accel. Reg\n(Recent)":     min(acc_fc_2030, 55000),
        "Likely\n(Estimated)":      38000,
    }
    sc_colors = [COLORS["linear"], COLORS["poly2"], COLORS["poly3"], COLORS["xgboost"]]
    bars = ax2.bar(range(4), scenarios.values(), color=sc_colors, width=0.55,
                   edgecolor="white", lw=1.5, zorder=3)
    for bar, (name, val) in zip(bars, scenarios.items()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                 f"{val:,}", ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(TARGET_KM, color="#c0392b", lw=2.2, ls="--",
                label=f"Target: {TARGET_KM:,} ckt km")
    ax2.fill_between([-0.5, 3.5], [0]*2, [TARGET_KM]*2,
                     color="#fadbd8", alpha=0.22)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["CEA Plan\n(Ideal)","Linear Reg\n(Trend)",
                          "Accel. Reg\n(Recent)","Likely\n(Estimated)"], fontsize=8)
    ax2.set_ylabel("Cumulative ckt km by 2030"); ax2.set_ylim(0, 62000)
    ax2.legend(fontsize=9)
    add_status_badge(ax2, "AT RISK")
    for i, (name, val) in enumerate(scenarios.items()):
        gap = TARGET_KM - val
        if gap > 0:
            ax2.text(i + 0.28, (val + TARGET_KM)/2,
                     f"−{gap:,.0f}\nkm", fontsize=7, color="#c0392b",
                     fontweight="bold", va="center")
    ax2.set_title("GOAL I-1 · 2030 Transmission Forecast\nvs CEA 50,890 ckt km Target")

    fig.suptitle("GOAL I-1 · ISTS Transmission Infrastructure · Target: 50,890 ckt km by 2030",
                 fontsize=14, fontweight="bold", color="#1a3a5c", y=1.02)

    return save_and_close(fig, "I1_transmission", plots_dir)


if __name__ == "__main__":
    run()
