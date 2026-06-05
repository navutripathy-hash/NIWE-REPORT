"""
goal_w1_wind_capacity.py
Goal W-1: Total Installed Wind Capacity — Target: 100 GW by FY 2030
Models: Polynomial Regression (deg 2 & 3) | ARIMA(1,1,1) | XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from utils import (apply_style, add_target_line, add_forecast_band,
                   add_status_badge, annotate_gap, save_and_close, COLORS)

TARGET_GW   = 100.0
FORECAST_YR = 2030
PLOTS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")


# ── Data ─────────────────────────────────────────────────────────────────────
FY_YEARS   = np.array([2007,2008,2009,2010,2011,2012,2013,2014,2015,
                        2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026])
CUMUL_GW   = np.array([7.85,9.587,10.925,13.064,16.084,18.421,20.150,22.465,
                        23.447,26.777,32.280,34.046,35.626,37.669,38.785,
                        40.355,42.633,45.887,50.017,56.100])
ANNUAL_ADD = np.array([0.0,1.737,1.338,2.139,3.020,2.337,1.729,2.315,0.982,
                        3.330,5.503,1.766,1.580,2.043,1.116,1.570,2.278,
                        3.254,4.130,6.083])

FORECAST_YEARS = np.arange(2027, 2031)
ALL_YEARS      = np.concatenate([FY_YEARS, FORECAST_YEARS])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Polynomial Regression (degree 2 & 3) on cumulative capacity
# ─────────────────────────────────────────────────────────────────────────────
def fit_poly_regression(degree: int):
    X = FY_YEARS.reshape(-1, 1).astype(float)
    y = CUMUL_GW
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr",   LinearRegression())
    ])
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    X_all  = ALL_YEARS.reshape(-1, 1).astype(float)
    y_pred = model.predict(X_all)
    fc_2030 = model.predict(np.array([[FORECAST_YR]]))[0]
    return y_pred, fc_2030, r2


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARIMA on annual additions → integrate to cumulative
# ─────────────────────────────────────────────────────────────────────────────
def fit_arima():
    additions = ANNUAL_ADD[1:]          # drop FY2007 (base)
    years_add = FY_YEARS[1:]

    model = ARIMA(additions, order=(1, 1, 1))
    result = model.fit()

    fc_obj     = result.get_forecast(steps=4)
    fc_mean    = fc_obj.predicted_mean
    fc_ci      = fc_obj.conf_int(alpha=0.25)  # ~75% CI — returns ndarray or DataFrame

    # Normalise conf_int to a plain numpy array (shape 4×2)
    if hasattr(fc_ci, "values"):          # pandas DataFrame
        fc_ci_arr = fc_ci.values
    else:                                 # already numpy ndarray
        fc_ci_arr = np.array(fc_ci)

    # Integrate: start from 2026 cumulative
    cum_base = CUMUL_GW[-1]
    cum_fc   = [cum_base]
    for a in fc_mean:
        cum_fc.append(cum_fc[-1] + max(a, 0))
    cum_forecast = np.array(cum_fc[1:])  # years 2027-2030

    lo = np.array([cum_base + max(float(fc_ci_arr[i, 0]), 0) * max(i+1, 1) * 0.75
                   for i in range(4)])
    hi = np.array([cum_base + abs(float(fc_ci_arr[i, 1])) * max(i+1, 1) * 0.75
                   for i in range(4)])

    return cum_forecast, fc_mean, years_add, result.fittedvalues, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost with lag features on annual additions
# ─────────────────────────────────────────────────────────────────────────────
def fit_xgboost():
    adds  = ANNUAL_ADD[1:]   # 19 values: FY2008–FY2026
    years = FY_YEARS[1:]     # 19 values: FY2008–FY2026

    # lag1[i] = adds[i+1], lag2[i] = adds[i], y[i] = adds[i+2]
    # So all three have len = len(adds) - 2 = 17
    lag1  = adds[1:-1]       # 17 values
    lag2  = adds[0:-2]       # 17 values
    y_xgb = adds[2:]         # 17 values
    yr_n  = (years[2:] - 2007) / (2030 - 2007)   # 17 values

    X_train = np.column_stack([yr_n, lag1, lag2])
    y_train = y_xgb

    xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.85, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    train_pred = xgb.predict(X_train)

    # Autoregressive forecast 2027-2030
    add_history = list(adds)
    cum_fc = [CUMUL_GW[-1]]
    annual_fc  = []
    for yr in FORECAST_YEARS:
        yr_norm = (yr - 2007) / (2030 - 2007)
        x_in    = np.array([[yr_norm, add_history[-1], add_history[-2]]])
        pred    = float(xgb.predict(x_in)[0])
        pred    = max(pred, 4.0)          # floor at 4 GW
        annual_fc.append(pred)
        add_history.append(pred)
        cum_fc.append(cum_fc[-1] + pred)

    cum_forecast = np.array(cum_fc[1:])
    return cum_forecast, train_pred, years[2:], annual_fc


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def run(plots_dir=PLOTS_DIR):
    apply_style()

    poly2_pred, poly2_fc30, r2_p2 = fit_poly_regression(2)
    poly3_pred, poly3_fc30, r2_p3 = fit_poly_regression(3)
    arima_cum, arima_ann, yr_add, arima_fit, arima_lo, arima_hi = fit_arima()
    xgb_cum, xgb_fit, yr_xgb, xgb_ann = fit_xgboost()

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#fafafa")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── Panel 1: Polynomial Regression ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(FY_YEARS, CUMUL_GW, color=COLORS["historical"], s=55,
                zorder=5, label="Actual (FY2007–FY2026)")
    ax1.plot(ALL_YEARS, poly2_pred, color=COLORS["poly2"], lw=2.2,
             label=f"Poly Deg-2 (R²={r2_p2:.3f})  →2030: {poly2_fc30:.1f} GW")
    ax1.plot(ALL_YEARS, poly3_pred, color=COLORS["poly3"], lw=2.2,
             linestyle="-.", label=f"Poly Deg-3 (R²={r2_p3:.3f})  →2030: {poly3_fc30:.1f} GW")
    ax1.axvline(2026.5, color="#aaa", lw=1, linestyle=":")
    ax1.text(2026.7, 10, "Forecast →", fontsize=8, color="#777")
    add_target_line(ax1, TARGET_GW, "Target: 100 GW")
    add_forecast_band(ax1, FORECAST_YEARS,
                      [min(poly2_fc30, poly3_fc30)] * 4,
                      [max(poly2_fc30, poly3_fc30)] * 4)
    ax1.set_title("GOAL W-1 · Polynomial Regression\n(Cumulative Wind Capacity)")
    ax1.set_xlabel("Financial Year"); ax1.set_ylabel("Cumulative Installed (GW)")
    ax1.legend(loc="upper left", fontsize=7.5)
    best_fc = max(poly2_fc30, poly3_fc30)
    status  = "AT RISK" if best_fc < TARGET_GW else "LIKELY ACHIEVED"
    add_status_badge(ax1, status)
    annotate_gap(ax1, 2030.3, best_fc, TARGET_GW)
    ax1.set_xlim(2006, 2031.5); ax1.set_ylim(0, 130)

    # ── Panel 2: ARIMA Forecast ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(yr_add, ANNUAL_ADD[1:], color=COLORS["historical"], alpha=0.5,
            width=0.6, label="Actual Annual Addition")
    ax2.plot(yr_add, arima_fit, color=COLORS["arima"], lw=1.8,
             label="ARIMA(1,1,1) Fitted")
    ax2.bar(FORECAST_YEARS, arima_ann, color=COLORS["arima"], alpha=0.65,
            width=0.6, label=f"ARIMA Forecast (GW/yr)")
    for yr, val in zip(FORECAST_YEARS, arima_ann):
        ax2.text(yr, val + 0.15, f"{val:.1f}", ha="center", fontsize=7.5,
                 color=COLORS["arima"], fontweight="bold")
    ax2.axvline(2026.5, color="#aaa", lw=1, linestyle=":")
    ax2.text(2026.7, 0.3, "Forecast →", fontsize=8, color="#777")
    ax2.axhline(y=11, color=COLORS["target"], lw=1.6, linestyle="--",
                label="Required ~11 GW/yr for 100 GW")
    arima_fc30 = arima_cum[-1]
    ax2.set_title(f"GOAL W-1 · ARIMA(1,1,1)\n(Annual Additions → 2030: {arima_fc30:.1f} GW cumul.)")
    ax2.set_xlabel("Financial Year"); ax2.set_ylabel("Annual Addition (GW/yr)")
    ax2.legend(loc="upper left", fontsize=7.5)
    add_status_badge(ax2, "AT RISK" if arima_fc30 < TARGET_GW else "LIKELY ACHIEVED")
    ax2.set_xlim(2006, 2032); ax2.set_ylim(0, 14)

    # ── Panel 3: XGBoost Forecast ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(FY_YEARS, CUMUL_GW, color=COLORS["historical"], s=55,
                zorder=5, label="Actual Cumulative")
    # Reconstruct XGBoost cumulative on training years
    cum_train = list(CUMUL_GW)
    ax3.plot(FY_YEARS, CUMUL_GW, color=COLORS["historical"], lw=1.5, alpha=0.5)
    ax3.plot(FORECAST_YEARS, xgb_cum, color=COLORS["xgboost"], lw=2.5,
             marker="o", ms=6, label=f"XGBoost Forecast → 2030: {xgb_cum[-1]:.1f} GW")
    # Connect last historical to first forecast
    ax3.plot([FY_YEARS[-1], FORECAST_YEARS[0]], [CUMUL_GW[-1], xgb_cum[0]],
             color=COLORS["xgboost"], lw=2.5, linestyle="--")
    add_forecast_band(ax3, FORECAST_YEARS,
                      xgb_cum * 0.93, xgb_cum * 1.07,
                      color=COLORS["xgboost"], alpha=0.12)
    ax3.axvline(2026.5, color="#aaa", lw=1, linestyle=":")
    ax3.text(2026.7, 10, "Forecast →", fontsize=8, color="#777")
    add_target_line(ax3, TARGET_GW, "Target: 100 GW")
    xgb_fc30 = xgb_cum[-1]
    ax3.set_title(f"GOAL W-1 · XGBoost (Lag Features)\n(Autoregressive Annual Add → 2030: {xgb_fc30:.1f} GW)")
    ax3.set_xlabel("Financial Year"); ax3.set_ylabel("Cumulative Installed (GW)")
    ax3.legend(loc="upper left", fontsize=7.5)
    add_status_badge(ax3, "AT RISK" if xgb_fc30 < TARGET_GW else "LIKELY ACHIEVED")
    annotate_gap(ax3, 2030.3, xgb_fc30, TARGET_GW)
    ax3.set_xlim(2006, 2031.5); ax3.set_ylim(0, 130)

    # ── Panel 4: Combined Model Comparison ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    fc_vals = {
        "Poly Deg-2": poly2_fc30,
        "Poly Deg-3": poly3_fc30,
        "ARIMA(1,1,1)": arima_fc30,
        "XGBoost": xgb_fc30,
    }
    bar_colors = [COLORS["poly2"], COLORS["poly3"], COLORS["arima"], COLORS["xgboost"]]
    bars = ax4.bar(fc_vals.keys(), fc_vals.values(), color=bar_colors, width=0.5,
                   edgecolor="white", linewidth=1.5, zorder=3)
    for bar, val in zip(bars, fc_vals.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{val:.1f} GW", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ax4.axhline(y=TARGET_GW, color=COLORS["target"], lw=2.2, linestyle="--",
                label=f"Target: {TARGET_GW} GW", zorder=4)
    ax4.fill_between([-0.5, 3.5], [TARGET_GW]*2, [130]*2,
                     color="#d5f5e3", alpha=0.35, label="Achievable zone")
    ax4.fill_between([-0.5, 3.5], [0]*2, [TARGET_GW]*2,
                     color="#fadbd8", alpha=0.25, label="Gap zone")
    ax4.set_title("GOAL W-1 · 2030 Forecast Comparison\n(All Models — Target 100 GW)")
    ax4.set_ylabel("Forecast 2030 Capacity (GW)")
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 130)
    # Annotate gap for each bar
    for bar, val in zip(bars, fc_vals.values()):
        if val < TARGET_GW:
            gap = TARGET_GW - val
            ax4.annotate(f"-{gap:.1f} GW",
                         xy=(bar.get_x() + bar.get_width()/2, TARGET_GW),
                         xytext=(bar.get_x() + bar.get_width()/2, (val + TARGET_GW)/2 + 3),
                         ha="center", fontsize=7.5, color="#c0392b",
                         arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    add_status_badge(ax4, "AT RISK")

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle("GOAL W-1 · India Total Installed Wind Capacity · Target: 100 GW by FY 2030",
                 fontsize=15, fontweight="bold", color="#1a3a5c", y=1.01)

    return save_and_close(fig, "W1_wind_capacity", plots_dir)


if __name__ == "__main__":
    run()
