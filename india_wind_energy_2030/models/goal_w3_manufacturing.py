"""
goal_w3_manufacturing.py
Goal W-3: Wind Manufacturing Localisation — Target: 85% by 2030
Models: Linear Regression | Polynomial (deg 2) Regression
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
                   add_status_badge, save_and_close, COLORS)

TARGET_PCT  = 85.0
FORECAST_YR = 2030
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

YEARS       = np.array([2022, 2023, 2024, 2025, 2026], dtype=float)
LOCAL_PCT   = np.array([55.0, 58.0, 64.0, 70.0, 78.0])   # localisation %
MFG_CAP     = np.array([10.0, 12.0, 18.0, 20.0, 24.0])   # GW/yr capacity
INSTALL_GW  = np.array([2.28,  3.25,  4.15,  6.30,  6.08])
UTIL_PCT    = INSTALL_GW / MFG_CAP * 100

FORECAST_YRS = np.arange(2027, 2031)
ALL_YRS      = np.concatenate([YEARS, FORECAST_YRS])


def fit_model(degree):
    X    = YEARS.reshape(-1, 1)
    y    = LOCAL_PCT
    pipe = Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)),
                     ("lr",   LinearRegression())])
    pipe.fit(X, y)
    r2   = r2_score(y, pipe.predict(X))
    pred = pipe.predict(ALL_YRS.reshape(-1, 1))
    fc30 = pipe.predict(np.array([[FORECAST_YR]]))[0]
    return pred, fc30, r2


def run(plots_dir=PLOTS_DIR):
    apply_style()

    lin_pred, lin_fc30, r2_lin = fit_model(1)
    pol_pred, pol_fc30, r2_pol = fit_model(2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#fafafa")

    # ── Panel 1: Linear Regression ────────────────────────────────────────────
    ax1 = axes[0]
    ax1.scatter(YEARS, LOCAL_PCT, color=COLORS["historical"], s=80, zorder=5,
                label="Actual Localisation %")
    ax1.plot(ALL_YRS, lin_pred, color=COLORS["linear"], lw=2.2,
             label=f"Linear Reg (R²={r2_lin:.3f})  →2030: {lin_fc30:.1f}%")
    ax1.axvline(2026.5, color="#aaa", lw=1, ls=":")
    ax1.text(2026.7, 55, "Forecast →", fontsize=8, color="#777")
    add_target_line(ax1, TARGET_PCT, f"Target: {TARGET_PCT}%")
    # CI band
    resid  = LOCAL_PCT - Pipeline([("poly", PolynomialFeatures(1,include_bias=False)),
                                    ("lr", LinearRegression())]).fit(
                                    YEARS.reshape(-1,1), LOCAL_PCT).predict(YEARS.reshape(-1,1))
    sigma = np.std(resid)
    add_forecast_band(ax1, ALL_YRS, lin_pred - 1.5*sigma, lin_pred + 1.5*sigma,
                      color=COLORS["linear"])
    for yr, val in zip(YEARS, LOCAL_PCT):
        ax1.annotate(f"{val}%", (yr, val), textcoords="offset points",
                     xytext=(0, 8), fontsize=8, ha="center", color=COLORS["historical"])
    status = "LIKELY ACHIEVED" if lin_fc30 >= 80 else "AT RISK"
    add_status_badge(ax1, status)
    ax1.set_title("GOAL W-3 · Linear Regression\n(Localisation % over time)")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Local Content (%)")
    ax1.legend(fontsize=8); ax1.set_ylim(40, 100); ax1.set_xlim(2021, 2031)

    # ── Panel 2: Polynomial Regression ────────────────────────────────────────
    ax2 = axes[1]
    ax2.scatter(YEARS, LOCAL_PCT, color=COLORS["historical"], s=80, zorder=5,
                label="Actual Localisation %")
    ax2.plot(ALL_YRS, pol_pred, color=COLORS["poly2"], lw=2.2,
             label=f"Poly Deg-2 (R²={r2_pol:.3f})  →2030: {pol_fc30:.1f}%")
    ax2.axvline(2026.5, color="#aaa", lw=1, ls=":")
    add_target_line(ax2, TARGET_PCT, f"Target: {TARGET_PCT}%")
    add_forecast_band(ax2, ALL_YRS, pol_pred - 3, pol_pred + 3,
                      color=COLORS["poly2"])
    status2 = "LIKELY ACHIEVED" if pol_fc30 >= 80 else "AT RISK"
    add_status_badge(ax2, status2)
    ax2.set_title("GOAL W-3 · Polynomial Regression (Deg-2)\n(Localisation % over time)")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Local Content (%)")
    ax2.legend(fontsize=8); ax2.set_ylim(40, 100); ax2.set_xlim(2021, 2031)

    # ── Panel 3: Manufacturing Utilisation Rate Trend ─────────────────────────
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    bars = ax3.bar(YEARS - 0.2, MFG_CAP, width=0.38, color=COLORS["poly3"],
                   alpha=0.7, label="Manufacturing Capacity (GW/yr)")
    bars2 = ax3.bar(YEARS + 0.2, INSTALL_GW, width=0.38, color=COLORS["historical"],
                    alpha=0.85, label="Actual Installation (GW)")
    ax3_twin.plot(YEARS, UTIL_PCT, color=COLORS["arima"], lw=2.2, marker="^",
                  ms=7, label="Utilisation Rate (%)")
    ax3_twin.set_ylabel("Utilisation Rate (%)", color=COLORS["arima"])
    ax3_twin.tick_params(axis='y', labelcolor=COLORS["arima"])
    ax3_twin.set_ylim(0, 60)
    ax3_twin.axhline(100, color=COLORS["arima"], lw=1, ls="--", alpha=0.3)

    ax3.set_title("GOAL W-3 · Manufacturing Capacity vs\nActual Installation + Utilisation Rate")
    ax3.set_xlabel("Year"); ax3.set_ylabel("GW")
    lines1, lbl1 = ax3.get_legend_handles_labels()
    lines2, lbl2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1+lines2, lbl1+lbl2, fontsize=7.5, loc="upper left")

    fig.suptitle("GOAL W-3 · Wind Manufacturing Localisation · Target: 85% by 2030",
                 fontsize=14, fontweight="bold", color="#1a3a5c", y=1.02)

    return save_and_close(fig, "W3_manufacturing", plots_dir)


if __name__ == "__main__":
    run()
