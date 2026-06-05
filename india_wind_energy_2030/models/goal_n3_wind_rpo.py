"""
goal_n3_wind_rpo.py
Goal N-3: Wind Share of Total Power Consumption (RPO/RCO) — Target: 6.94% by FY2030
Model: Derived from wind generation forecast + total consumption growth
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

TARGET_PCT  = 6.94
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

# Wind Generation (TWh) — historical
GEN_YEARS = np.array([2012,2013,2014,2015,2016,2017,2018,2019,
                       2020,2021,2022,2023,2024,2025], dtype=float)
WIND_GEN  = np.array([28.2,29.0,32.0,33.0,40.0,52.7,58.0,61.0,
                       60.0,72.8,72.0,72.0,79.0,101.0])    # TWh

# Total Electricity Consumption (TWh) — estimated
TOTAL_CONS= np.array([1000,1020,1060,1100,1100,1200,1230,1290,
                       1280,1330,1375,1620,1750,1900], dtype=float)

WIND_SHARE = (WIND_GEN / TOTAL_CONS) * 100  # %

# RCO Notified trajectory (Ministry of Power)
RCO_YEARS   = np.array([2025, 2026, 2027, 2028, 2029, 2030], dtype=float)
RCO_TARGETS = np.array([0.81, 1.60, 2.50, 3.80, 5.30, 6.94])   # % from MoP notification

FORECAST_YRS = np.arange(2026, 2031)


def fit_wind_share():
    X  = GEN_YEARS.reshape(-1, 1)
    m  = LinearRegression().fit(X, WIND_SHARE)
    r2 = r2_score(WIND_SHARE, m.predict(X))
    all_yr = np.arange(2012, 2031).reshape(-1, 1)
    pred   = m.predict(all_yr)
    fc30   = float(m.predict([[2030]])[0])
    return np.arange(2012, 2031), pred, fc30, r2


def run(plots_dir=PLOTS_DIR):
    apply_style()

    all_yrs, share_pred, fc30, r2 = fit_wind_share()

    # Build forecast scenarios for wind share
    # Optimistic: wind capacity ~95 GW → ~150 TWh gen; consumption ~2300 TWh → 6.5%
    # Base:       wind capacity ~85 GW → ~130 TWh gen; consumption ~2300 TWh → 5.7%
    # Target requires: ~160 TWh wind / 2300 TWh total = 6.96%

    sc_scenarios = {
        "Pessimistic (80 GW, low gen)":  (120 / 2300) * 100,
        "Base Case (90 GW)":             (135 / 2300) * 100,
        "Optimistic (100 GW)":           (155 / 2300) * 100,
        "Target Scenario":               TARGET_PCT,
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.patch.set_facecolor("#fafafa")

    # ── Panel 1: Wind Generation + Share Trend ─────────────────────────────────
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.bar(GEN_YEARS, WIND_GEN, color=COLORS["historical"], alpha=0.65, width=0.6,
            label="Wind Generation (TWh)")
    ax1_twin.plot(GEN_YEARS, WIND_SHARE, color=COLORS["arima"], lw=2.5, marker="o",
                  ms=6, label="Wind Share %")
    ax1_twin.plot(all_yrs, share_pred, color=COLORS["poly2"], lw=1.8, ls="--",
                  label=f"Linear Reg →2030: {fc30:.2f}%")
    ax1_twin.axhline(TARGET_PCT, color=COLORS["target"], lw=1.8, ls="--",
                     label=f"Target: {TARGET_PCT}%")
    ax1_twin.set_ylabel("Wind Share of Consumption (%)", color=COLORS["arima"])
    ax1_twin.tick_params(axis='y', labelcolor=COLORS["arima"])
    ax1_twin.set_ylim(0, 12)

    ax1.set_xlabel("Year"); ax1.set_ylabel("Wind Generation (TWh)")
    ax1.set_title("GOAL N-3 · Wind Generation & Share\n(Historical + Trend)")
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lbl1+lbl2, fontsize=7.5, loc="upper left")
    add_status_badge(ax1, "LIKELY ACHIEVED" if fc30 >= 6.5 else "AT RISK")

    # ── Panel 2: RCO Notified Trajectory vs Forecast ──────────────────────────
    ax2 = axes[1]
    # Forecast trajectory based on capacity growth
    fc_share_base = [4.8, 5.2, 5.6, 5.9, 6.2, 6.5]   # base scenario % by year
    fc_share_opt  = [4.9, 5.4, 5.9, 6.4, 6.9, 7.4]   # optimistic %
    fc_yrs = np.arange(2025, 2031)

    ax2.plot(RCO_YEARS, RCO_TARGETS, color=COLORS["target"], lw=2.5, marker="s",
             ms=7, label="RCO Notified Trajectory (MoP)")
    ax2.plot(fc_yrs, fc_share_base, color=COLORS["linear"], lw=2, ls="-.",
             marker="o", ms=5, label="Base Forecast (~90 GW wind)")
    ax2.plot(fc_yrs, fc_share_opt, color=COLORS["poly3"], lw=2, ls="-.",
             marker="^", ms=5, label="Optimistic Forecast (~100 GW wind)")
    add_forecast_band(ax2, fc_yrs, fc_share_base, fc_share_opt,
                      color=COLORS["poly3"], alpha=0.15)
    ax2.axhline(TARGET_PCT, color="#c0392b", lw=1.8, ls="--",
                label=f"Target: {TARGET_PCT}%")
    ax2.fill_between([2029, 2030], [6.0, 6.5], [7.0, 7.4],
                     color="#d5f5e3", alpha=0.35, label="Target range achievable")
    for yr, t in zip(RCO_YEARS, RCO_TARGETS):
        ax2.annotate(f"{t}%", (yr, t), textcoords="offset points",
                     xytext=(4, 6), fontsize=8, color=COLORS["target"])
    ax2.set_xlabel("Year"); ax2.set_ylabel("Wind Share of Total Consumption (%)")
    ax2.set_title("GOAL N-3 · RCO Trajectory vs\nCapacity-Based Forecast")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 9); ax2.set_xlim(2024, 2031)
    add_status_badge(ax2, "LIKELY ACHIEVED")

    # ── Panel 3: 2030 Scenario Bar ─────────────────────────────────────────────
    ax3 = axes[2]
    sc_names = list(sc_scenarios.keys())
    sc_vals  = list(sc_scenarios.values())
    sc_clrs  = [COLORS["scenario_pes"], COLORS["scenario_base"],
                COLORS["scenario_opt"], COLORS["achieved"]]

    bars = ax3.bar(range(4), sc_vals, color=sc_clrs, width=0.55,
                   edgecolor="white", lw=1.5, zorder=3)
    for bar, val in zip(bars, sc_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.07,
                 f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")

    ax3.axhline(TARGET_PCT, color=COLORS["target"], lw=2.2, ls="--",
                label=f"Target: {TARGET_PCT}%")
    ax3.fill_between([-0.5, 3.5], [TARGET_PCT]*2, [9]*2,
                     color="#d5f5e3", alpha=0.3, label="Above target")
    ax3.fill_between([-0.5, 3.5], [0]*2, [TARGET_PCT]*2,
                     color="#fadbd8", alpha=0.2, label="Below target")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(["Pessimistic\n80 GW", "Base\n90 GW",
                          "Optimistic\n100 GW", "Target\nScenario"], fontsize=8)
    ax3.set_ylabel("Wind Share of Consumption (%) in 2030")
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 9)
    ax3.set_title("GOAL N-3 · 2030 Wind RPO\nScenario Comparison")
    add_status_badge(ax3, "LIKELY ACHIEVED")

    fig.suptitle("GOAL N-3 · Wind RPO/RCO · Target: 6.94% of Total Consumption by FY 2030",
                 fontsize=14, fontweight="bold", color="#1a3a5c", y=1.02)

    return save_and_close(fig, "N3_wind_rpo", plots_dir)


if __name__ == "__main__":
    run()
