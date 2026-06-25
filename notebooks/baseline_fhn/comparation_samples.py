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
        Path("results/fhn_obs/samples").glob("results_202606*.pkl")
    ),
    "fhn_dyn": next(
        Path("results/fhn/fhn_samples").glob("results_202606*.pkl")
    ),
}

records = []

for signal, result_file in files.items():

    with open(result_file, "rb") as f:
        all_results = pickle.load(f)

    for entry in all_results:

        samples = entry["samples"]

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
                        "samples": samples,
                        "Method": method,
                        "AUC": auc,
                    }
                )

# --------------------------------------------------
# DataFrame
# --------------------------------------------------

df_results = pd.DataFrame(records)

df_results["samples"] = df_results["samples"].astype(int)

# --------------------------------------------------
# Group statistics
# --------------------------------------------------

df_grouped = (
    df_results
    .groupby(["Signal", "Method", "samples"])
    .agg(
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
    )
    .reset_index()
)

# --------------------------------------------------
# Style
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
            .sort_values("samples")
        )

        plt.errorbar(
            data["samples"],
            data["AUC_mean"],
            yerr=data["AUC_std"],
            fmt=marker,
            color=color,
            capsize=5,
            linestyle="None",
        )

# --------------------------------------------------
# Axes formatting
# --------------------------------------------------

plt.xlabel(r"Samples ($N_s$)")
plt.ylabel("AUC")

plt.grid(True, linestyle="--", alpha=0.4)

plt.ylim(0.0, 1.1)
#plt.xlim(0, 255)

# --------------------------------------------------
# FIXED xticks (no overlap)
# --------------------------------------------------

xticks = sorted(df_grouped["samples"].unique())
xticks = xticks[::3]

plt.xticks(xticks)
plt.xlim(0,255)

# --------------------------------------------------
# Panel label
# --------------------------------------------------

plt.text(
    -0.13,
    1.01,
    "(e)",
    fontweight="bold",
    fontsize=13,
    va="bottom",
    ha="left",
    transform=plt.gca().transAxes,
)

plt.tight_layout()

# --------------------------------------------------
# Save
# --------------------------------------------------

plt.savefig(
    "fhn_samples_combined.eps",
    format="eps",
    dpi=180,
)

plt.show()