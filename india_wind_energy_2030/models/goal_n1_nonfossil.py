"""
goal_n1_nonfossil.py
Goal N-1: Total Non-Fossil Fuel Installed Capacity — Target: 500 GW by 2030
Models: Linear Regression | Polynomial (deg 2) | Scenario Extrapolation
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from utils import (apply_style, add_target_line, add_forecast_band,
                   add_status_badge, annotate_gap, save_and_close, COLORS)

TARGET_GW   = 500.0
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

YEARS       = np.array([2019,2020,2021,2022,2023,2024,2025,2026], dtype=float)
NON_FOSSIL  = np.array([134, 145, 157, 175, 198, 220, 245, 266], dtype=float)
ANNUAL_ADD  = np.diff(NON_FOSSIL)

FORECAST_YRS = np.arange(2027, 2031)
ALL_YRS      = np.concatenate([YEARS, FORECAST_YRS])


def fit_model(degree):
    X    = YEARS.reshape(-1, 1)
    pipe = Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)),
                     ("lr",   LinearRegression())])
    pipe.fit(X, NON_FOSSIL)
    r2   = r2_score(NON_FOSSIL, pipe.predict(X))
    pred = pipe.predict(ALL_YRS.reshape(-1, 1))
    fc30 = float(pipe.predict(np.array([[2030]]))[0])
    return pred, fc30, r2


def run(plots_dir=PLOTS_DIR):
    apply_style()

    lin_pred, lin_fc30, r2_lin = fit_model(1)
    pol_pred, pol_fc30, r2_pol = fit_model(2)

    # Scenario projections
    scenarios = {
        "Pessimistic (30 GW/yr)":  NON_FOSSIL[-1] + 30 * 4,   # 266 + 120 = 386
        "Base Case (40 GW/yr)":    NON_FOSSIL[-1] + 40 * 4,   # 266 + 160 = 426
        "Optimistic (50 GW/yr)":   NON_FOSSIL[-1] + 50 * 4,   # 266 + 200 = 466
        "Required (58 GW/yr)":     NON_FOSSIL[-1] + 58 * 4,   # 266 + 232 ≈ 498
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.patch.set_facecolor("#fafafa")

    # ── Panel 1: Regression Models ────────────────────────────────────────────
    ax1 = axes[0]
    ax1.scatter(YEARS, NON_FOSSIL, color=COLORS["historical"], s=65, zorder=5,
                label="Actual Non-Fossil Capacity (GW)")
    ax1.plot(ALL_YRS, lin_pred, color=COLORS["linear"], lw=2.2,
             label=f"Linear Reg (R²={r2_lin:.4f})  →2030: {lin_fc30:.0f} GW")
    ax1.plot(ALL_YRS, pol_pred, color=COLORS["poly2"], lw=2.2, ls="-.",
             label=f"Poly Deg-2 (R²={r2_pol:.4f})  →2030: {pol_fc30:.0f} GW")
    ax1.axvline(2026.5, color="#aaa", lw=1, ls=":")
    ax1.text(2026.7, 130, "Forecast →", fontsize=8, color="#777")
    add_target_line(ax1, TARGET_GW, "Target: 500 GW")
    add_forecast_band(ax1, FORECAST_YRS,
                      [min(lin_fc30, pol_fc30)] * 4,
                      [max(lin_fc30, pol_fc30)] * 4)
    for yr, val in zip(YEARS, NON_FOSSIL):
        ax1.annotate(f"{val:.0f}", (yr, val), textcoords="offset points",
                     xytext=(3, 6), fontsize=7.5, color=COLORS["historical"])
    best_fc = max(lin_fc30, pol_fc30)
    annotate_gap(ax1, 2030.3, best_fc, TARGET_GW)
    add_status_badge(ax1, "AT RISK")
    ax1.set_title("GOAL N-1 · Regression Models\n(Non-Fossil Capacity vs Year)")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Installed Non-Fossil Capacity (GW)")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.set_ylim(0, 560); ax1.set_xlim(2018, 2031.5)

    # ── Panel 2: Annual Additions Trend + Required Rate ────────────────────────
    ax2 = axes[1]
    add_years = YEARS[1:]
    ax2.bar(add_years - 0.2, ANNUAL_ADD, width=0.4, color=COLORS["historical"],
            alpha=0.8, label="Annual Addition (GW/yr)")
    # Required rate
    req_rate = (TARGET_GW - NON_FOSSIL[-1]) / 4
    ax2.axhline(req_rate, color="#c0392b", lw=2, ls="--",
                label=f"Required rate: {req_rate:.0f} GW/yr")
    ax2.axhline(np.mean(ANNUAL_ADD[-3:]), color=COLORS["linear"], lw=1.8, ls="-.",
                label=f"Avg last 3 yrs: {np.mean(ANNUAL_ADD[-3:]):.0f} GW/yr")
    # Scenario bars
    scenario_rates = [30, 40, 50, 58]
    for i, (yr, rate) in enumerate(zip(FORECAST_YRS, [40, 43, 47, 52])):
        ax2.bar(yr + 0.2, rate, width=0.4, color=COLORS["poly2"], alpha=0.65)
    ax2.text(2028.5, 55, f"Required: {req_rate:.0f} GW/yr", color="#c0392b",
             fontsize=8, fontweight="bold")
    ax2.set_title("GOAL N-1 · Annual Addition Rate\nvs Required Rate for 500 GW")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Annual Addition (GW/yr)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 75)
    add_status_badge(ax2, "AT RISK")

    # ── Panel 3: 2030 Scenario Comparison ─────────────────────────────────────
    ax3 = axes[2]
    sc_labels = list(scenarios.keys())
    sc_vals   = list(scenarios.values())
    sc_colors = [COLORS["scenario_pes"], COLORS["scenario_base"],
                 COLORS["scenario_opt"], COLORS["linear"]]

    bars = ax3.bar(range(len(sc_labels)), sc_vals, color=sc_colors, width=0.55,
                   edgecolor="white", lw=1.5, zorder=3)
    for bar, val in zip(bars, sc_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"{val:.0f} GW", ha="center", fontsize=9.5, fontweight="bold")

    ax3.axhline(TARGET_GW, color=COLORS["target"], lw=2.2, ls="--",
                label="Target: 500 GW")
    ax3.fill_between([-0.5, 3.5], [TARGET_GW]*2, [560]*2,
                     color="#d5f5e3", alpha=0.3, label="Achievable zone")
    ax3.fill_between([-0.5, 3.5], [0]*2, [TARGET_GW]*2,
                     color="#fadbd8", alpha=0.22, label="Gap zone")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(["Pessimistic\n(30 GW/yr)", "Base\n(40 GW/yr)",
                          "Optimistic\n(50 GW/yr)", "Required\n(58 GW/yr)"], fontsize=8)
    ax3.set_ylabel("Forecast 2030 Non-Fossil Capacity (GW)")
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 560)
    for i, val in enumerate(sc_vals):
        gap = TARGET_GW - val
        if gap > 0:
            ax3.text(i + 0.28, (val + TARGET_GW) / 2,
                     f"−{gap:.0f} GW", fontsize=7.5, color="#c0392b",
                     fontweight="bold", va="center")
    ax3.set_title("GOAL N-1 · 2030 Scenario Forecasts\nvs 500 GW Target")
    add_status_badge(ax3, "AT RISK")

    fig.suptitle("GOAL N-1 · India Total Non-Fossil Installed Capacity · Target: 500 GW by 2030",
                 fontsize=14, fontweight="bold", color="#1a3a5c", y=1.02)

    return save_and_close(fig, "N1_nonfossil", plots_dir)


if __name__ == "__main__":
    run()
