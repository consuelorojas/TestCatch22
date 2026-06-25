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
        Path("results/fhn_obs/noise").glob("results_20260616*.pkl")
    ),
    "fhn_dyn": next(
        Path("results/fhn/fhn_noise").glob("results_20260616*.pkl")
    ),
}

records = []

for signal, result_file in files.items():

    with open(result_file, "rb") as f:
        all_results = pickle.load(f)

    for entry in all_results:

        noise = entry["noise"]

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
                        "noise": noise,
                        "Method": method,
                        "AUC": auc,
                    }
                )

# --------------------------------------------------
# Convert to DataFrame
# --------------------------------------------------

df_results = pd.DataFrame(records)

# --------------------------------------------------
# Compute mean and std
# --------------------------------------------------

df_grouped = (
    df_results
    .groupby(["Signal", "Method", "noise"])
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

        data = df_grouped[
            (df_grouped["Method"] == method)
            & (df_grouped["Signal"] == signal)
        ].sort_values("noise")

        plt.errorbar(
            data["noise"],
            data["AUC_mean"],
            yerr=data["AUC_std"],
            fmt=marker,
            color=color,
            capsize=5,
            linestyle="None",
        )

# --------------------------------------------------
# Legends
# --------------------------------------------------
'''
method_handles = [
    plt.Line2D(
        [0], [0],
        color=color,
        marker="o",
        linestyle="None",
        label=method,
    )
    for method, color in method_colors.items()
]

legend1 = plt.legend(
    handles=method_handles,
    title="Method",
    loc="lower left",
    ncol=2,
)

signal_handles = [
    plt.Line2D(
        [0], [0],
        color="black",
        marker="*",
        linestyle="None",
        markersize=10,
        label="FHN Obs",
    ),
    plt.Line2D(
        [0], [0],
        color="black",
        marker="s",
        linestyle="None",
        markersize=8,
        label="FHN Dyn",
    ),
]

legend2 = plt.legend(
    handles=signal_handles,
    title="Signal",
    loc="lower right",
)

plt.gca().add_artist(legend1)
'''

# --------------------------------------------------
# Axes formatting
# --------------------------------------------------

plt.xlabel(r"Noise strength $(D)$")
plt.ylabel("AUC")

plt.grid(True)
plt.ylim(0.5, 1.05)

xticks = sorted(df_grouped["noise"].unique())
plt.xticks(xticks[::3])

plt.text(
    -0.13,
    1.01,
    "(d)",
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
    "noise_fhn_combined.eps",
    format="eps",
    dpi=180,
)

plt.show()