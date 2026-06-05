# 🌬️ India Wind Energy 2030 — ML Forecasting Project

> **"Will India achieve its wind energy goals by 2030?"**  
> A data-driven research project using historical data (2006–2026) and ML models to forecast India's wind energy trajectory.

---

## 📊 Key Findings

| Goal | Target | Forecast 2030 | Status |
|------|--------|--------------|--------|
| Total Wind Capacity | 100 GW | ~85–95 GW | ⚠️ AT RISK |
| Offshore Wind | 30 GW | ~0–2 GW | ❌ NOT ACHIEVED |
| Manufacturing Localisation | 85% | ~80–82% | ✅ LIKELY ACHIEVED |
| ISTS Transmission | 50,890 ckt km | ~30–40K ckt km | ⚠️ AT RISK |
| Total Non-Fossil Capacity | 500 GW | ~380–450 GW | ⚠️ AT RISK |
| Non-Fossil Share | 50% | **52% (Achieved 2025!)** | ✅ ACHIEVED |
| Wind RPO/RCO | 6.94% | ~6.0–7.4% | ✅ LIKELY ACHIEVED |

---

## 🗂️ Project Structure

```
india_wind_energy_2030/
├── run_all.py                        # ← Run this to generate all plots
├── utils.py                          # Shared styling and helpers
├── requirements.txt
├── data/
│   ├── wind_capacity.csv             # FY2007–FY2026 capacity data
│   ├── wind_generation.csv           # Annual generation (GWh)
│   ├── offshore_wind.csv             # Offshore status history
│   ├── manufacturing.csv             # Manufacturing capacity & utilisation
│   ├── non_fossil_capacity.csv       # Total non-fossil installed GW
│   └── ists_transmission.csv         # Transmission line milestones
├── models/
│   ├── goal_w1_wind_capacity.py      # Poly Reg + ARIMA + XGBoost
│   ├── goal_w2_offshore.py           # Scenario analysis
│   ├── goal_w3_manufacturing.py      # Linear + Poly Regression
│   ├── goal_i1_transmission.py       # Linear Regression
│   ├── goal_n1_nonfossil.py          # Linear + Poly + Scenarios
│   ├── goal_n2_nonfossil_share.py    # Milestone (ACHIEVED)
│   ├── goal_n3_wind_rpo.py           # Generation-based forecast
│   └── goal_dashboard.py             # Master dashboard
└── plots/                            # Auto-generated PNG plots
```

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/india-wind-energy-2030.git
cd india-wind-energy-2030

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all models and generate plots
python run_all.py
```

Plots are saved to the `plots/` folder.

---

## 🤖 ML Models Used

| Goal | Models |
|------|--------|
| W-1 Total Wind Capacity | Polynomial Regression (deg 2 & 3) + ARIMA(1,1,1) + XGBoost |
| W-2 Offshore Wind | Scenario Analysis (no historical data) |
| W-3 Manufacturing | Linear Regression + Polynomial (deg 2) |
| I-1 Transmission | Linear Regression (actual vs planned) |
| N-1 Non-Fossil Total | Linear Regression + Polynomial + Scenarios |
| N-2 Non-Fossil Share | Milestone visualization (already achieved) |
| N-3 Wind RPO/RCO | Generation-capacity derived forecast |

---

## 📁 Data Sources

All data sourced from official government publications only:

- **MNRE** — Ministry of New and Renewable Energy (PIB press releases)
- **CEA** — Central Electricity Authority (Annual Reports, Transmission Plan 2022)
- **Ministry of Power** — RPO/RCO Notifications
- **UNFCCC** — India's Updated NDC (August 2022)
- **NIWE** — National Institute of Wind Energy
- **Ember** — Global Electricity Review 2026
- **GWEC** — India Wind Energy Market Outlook 2025
- **WWEA** — Annual Report 2025

---

## 📈 Sample Outputs

After running `run_all.py`, you'll find in `plots/`:

- `goal_W1_wind_capacity.png` — 4-panel: Poly Reg, ARIMA, XGBoost, Comparison
- `goal_W2_offshore_wind.png` — Scenario trajectories + bar chart
- `goal_W3_manufacturing.png` — Linear + Poly regression + utilisation
- `goal_I1_transmission.png` — Actual vs planned + gap chart
- `goal_N1_nonfossil.png` — Regression + rate analysis + scenarios
- `goal_N2_nonfossil_share.png` — Achievement milestone chart
- `goal_N3_wind_rpo.png` — RPO trajectory vs forecast
- `goal_00_master_dashboard.png` — All goals at a glance

---

## 📋 Research Context

This project is part of a larger research report:  
**"Will India Achieve Its Wind Energy Goals by 2030?"**

Phases covered: Goal identification → Data collection → Validation →
ML readiness → Forecasting → Gap analysis → Recommendations

---

## 📜 License

MIT License — free to use for research and educational purposes.
