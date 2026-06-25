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
        Path("results/fhn_obs/npoints").glob("results_20260623*.pkl")
    ),
    "fhn_dyn": next(
        Path("results/fhn/fhn_npp").glob("results_202606*.pkl")
    ),
}

records = []

for signal, result_file in files.items():

    with open(result_file, "rb") as f:
        all_results = pickle.load(f)

    for entry in all_results:

        npp = entry["npp"]

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
                        "npp": npp,
                        "Method": method,
                        "AUC": auc,
                    }
                )

# --------------------------------------------------
# DataFrame
# --------------------------------------------------

df_results = pd.DataFrame(records)

df_results["npp"] = df_results["npp"].astype(int)

# --------------------------------------------------
# Group stats
# --------------------------------------------------

df_grouped = (
    df_results
    .groupby(["Signal", "Method", "npp"])
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
            .sort_values("npp")
        )

        plt.errorbar(
            data["npp"],
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

plt.xlabel(r"Number of points per period $(N_{pp})$")
plt.ylabel("AUC")

plt.grid(True, linestyle="--", alpha=0.4)

plt.ylim(0.4, 1.05)

# --------------------------------------------------
# Clean xticks (avoid overlap)
# --------------------------------------------------

xticks = sorted(df_grouped["npp"].unique())
xticks = xticks[::2]

plt.xticks(xticks)

# --------------------------------------------------
# Panel label
# --------------------------------------------------

plt.text(
    -0.13,
    1.01,
    "(c)",
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
    "fhn_npp_combined.eps",
    format="eps",
    dpi=180,
)

plt.show()