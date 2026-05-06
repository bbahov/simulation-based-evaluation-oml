# plot_comparison.py
#
# Reads the CSV outputs from both evaluation scripts and generates
# a single overlay comparison figure suitable for the paper.
#
# Run after both evaluation scripts have completed:
#   python plot_comparison.py

import pandas as pd
import matplotlib.pyplot as plt

DRIFT_POINT = 2000
SMOOTH_WINDOW = 100  # re-smooth for cleaner paper figure

df_active = pd.read_csv("results/active_results.csv")
df_passive = pd.read_csv("results/passive_results.csv")

# Re-compute rolling accuracy at the wider window
active_smooth = df_active["correct"].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
passive_smooth = df_passive["correct"].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    df_active["step"], active_smooth,
    color="#2ca02c", linewidth=2, alpha=0.9,
    label="Active Evaluation (model selects items)",
)
ax.plot(
    df_passive["step"], passive_smooth,
    color="#1f77b4", linewidth=2, alpha=0.9,
    label="Passive Evaluation (random items)",
)
ax.axvline(
    x=DRIFT_POINT, color="#d62728", linestyle="--",
    linewidth=1.5, label="Concept Drift",
)

ax.set_title(
    "Recommendation Accuracy Under Concept Drift:\nActive vs. Passive Evaluation",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Step (user interactions)", fontsize=11)
ax.set_ylabel(f"Rolling Accuracy (window = {SMOOTH_WINDOW})", fontsize=11)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
plt.savefig("results/comparison_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("Plot saved to results/comparison_plot.png")
