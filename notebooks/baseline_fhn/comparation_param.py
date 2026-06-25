import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("report.mplstyle")

# --------------------------------------------------
# Load both result files
# --------------------------------------------------

files = {
    "fhn_obs": next(
        Path("results/fhn_obs/parameter").glob("results_20260624*.pkl")
    ),
    "fhn_dyn": next(
        Path("results/fhn/fhn_parameter").glob("results_20260616*.pkl")
    ),
}

records = []

for signal, result_file in files.items():

    with open(result_file, "rb") as f:
        all_results = pickle.load(f)

    for entry in all_results:

        b = entry["b"]

        for method in [
            "raw",
            "pca",
            "fft",
            "fft_pca",
            "features",
            "features_pca",
        ]:

            for auc in entry[method]:

                records.append(
                    {
                        "Signal": signal,
                        "b": b,
                        "Method": method,
                        "AUC": auc,
                    }
                )

# --------------------------------------------------
# Convert to DataFrame
# --------------------------------------------------

df_results = pd.DataFrame(records)

# FIX: avoid float noise in grouping / ticks
df_results["b"] = df_results["b"].round(4)

# --------------------------------------------------
# Compute mean and std
# --------------------------------------------------

df_grouped = (
    df_results
    .groupby(["Signal", "Method", "b"])
    .agg(
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
    )
    .reset_index()
)

# --------------------------------------------------
# Plot settings
# --------------------------------------------------

method_colors = {
    "raw": "C0",
    "pca": "C1",
    "fft": "C2",
    "fft_pca": "C3",
    "features": "C4",
    "features_pca": "C5",
}

signal_markers = {
    "fhn_obs": "*",
    "fhn_dyn": "s",
}

plt.figure(figsize=(6.4, 4.8))

# --------------------------------------------------
# Plot
# --------------------------------------------------

for method, color in method_colors.items():

    for signal, marker in signal_markers.items():

        data = (
            df_grouped[
                (df_grouped["Method"] == method)
                & (df_grouped["Signal"] == signal)
            ]
            .sort_values("b")
        )

        plt.errorbar(
            data["b"],
            data["AUC_mean"],
            yerr=data["AUC_std"],
            color=color,
            marker=marker,
            linestyle="None",
            #linestyle="-",
            #linewidth=1.2,
            #markersize=6,
            #capsize=2,
            alpha=0.85,
        )

# --------------------------------------------------
# Axes formatting
# --------------------------------------------------

plt.xlabel(r"Parameter difference $(b - b_0)$")
plt.ylabel("AUC")

plt.grid(True, which="both", alpha=0.4)

# safer limits (avoid clipping + ensure grid continuity)
b_min = df_grouped["b"].min()
b_max = df_grouped["b"].max()
plt.xlim(b_min - 0.01, b_max + 0.01)

plt.ylim(0.2, 1.05)

# --------------------------------------------------
# FIXED xticks (no overlap)
# --------------------------------------------------

xticks = np.linspace(b_min, b_max, 6)
plt.xticks(xticks, [f"{x:.2f}" for x in xticks])

# --------------------------------------------------
# Panel label (ONLY ONE)
# --------------------------------------------------

plt.text(
    -0.13,
    1.01,
    "(a)",
    fontweight="bold",
    fontsize=14,
    va="bottom",
    ha="left",
    transform=plt.gca().transAxes,
)

plt.tight_layout()

# --------------------------------------------------
# Save
# --------------------------------------------------

plt.savefig(
    "parameter_fhn_combined.eps",
    format="eps",
    dpi=180,
)

plt.show()