"""
run_all.py
Master runner: Executes all ML models and generates all plots.
Run from the project root:  python run_all.py
"""

import os
import sys
import time

# Ensure the project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Import all models
from models.goal_w1_wind_capacity   import run as run_w1
from models.goal_w2_offshore        import run as run_w2
from models.goal_w3_manufacturing   import run as run_w3
from models.goal_i1_transmission    import run as run_i1
from models.goal_n1_nonfossil       import run as run_n1
from models.goal_n2_nonfossil_share import run as run_n2
from models.goal_n3_wind_rpo        import run as run_n3
from models.goal_dashboard          import run as run_dash

MODELS = [
    ("W-1  Total Wind Capacity        (Poly Reg + ARIMA + XGBoost)", run_w1),
    ("W-2  Offshore Wind              (Scenario Analysis)",           run_w2),
    ("W-3  Manufacturing Localisation (Linear + Poly Reg)",           run_w3),
    ("I-1  ISTS Transmission          (Linear Reg)",                  run_i1),
    ("N-1  Total Non-Fossil Capacity  (Linear + Poly + Scenarios)",   run_n1),
    ("N-2  Non-Fossil Share           (Achieved — Milestone plot)",   run_n2),
    ("N-3  Wind RPO/RCO               (Generation Forecast)",         run_n3),
    ("     Master Dashboard           (All goals at a glance)",       run_dash),
]

SEP = "─" * 65

def main():
    print(f"\n{'='*65}")
    print("  INDIA WIND ENERGY 2030 — ML MODEL RUNNER")
    print(f"{'='*65}")
    print(f"  Output directory: {PLOTS_DIR}\n")

    saved = []
    t_start = time.time()

    for label, fn in MODELS:
        print(f"\n{SEP}")
        print(f"  ▶ Running: {label}")
        print(SEP)
        try:
            path = fn(plots_dir=PLOTS_DIR)
            saved.append(path)
        except Exception as e:
            print(f"  ✘ ERROR in {label}: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  ✔ All models complete in {elapsed:.1f}s")
    print(f"  ✔ {len(saved)} plots saved to: {PLOTS_DIR}")
    print(f"{'='*65}\n")
    print("  Generated files:")
    for p in saved:
        print(f"    • {os.path.basename(p)}")
    print()


if __name__ == "__main__":
    main()
