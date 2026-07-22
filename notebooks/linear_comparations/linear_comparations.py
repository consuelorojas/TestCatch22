import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ----------------------------------
#  load results 
# ----------------------------------

linear_file = Path("results/sine/sine_frequency/results_lineal_20260701_124831.pkl")
rbf_file = Path("results/sine/sine_frequency/results_20260610_095517.pkl")

with open(linear_file, "rb") as f:
    linear_results = pickle.load(f)

with open(rbf_file, "rb") as f:
    rbf_results = pickle.load(f)


# ----------------------------------
# convert to dataframe
# ----------------------------------

def results_to_dataframe(results, kernel):

    records = []

    for entry in results:
        df_val = entry["df"]

        for method in [
            "raw",
            "pca",
            "fft",
            "fft_pca",
            "features",
            "features_pca",
        ]:

            for auc in entry[method]:
                records.append({
                    "df": df_val,
                    "Method": method,
                    "Kernel": kernel,
                    "AUC": auc
                })

    return pd.DataFrame(records)


df_results = pd.concat(
    [
        results_to_dataframe(linear_results, "Linear"),
        results_to_dataframe(rbf_results, "RBF"),
    ],
    ignore_index=True
)

# -----------------------------
# Aggregate statistics
# -----------------------------
df_grouped = (
    df_results
    .groupby(["Kernel", "Method", "df"])
    .agg(
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std")
    )
    .reset_index()
)


# -----------------------------
# Style dictionaries
# -----------------------------
method_colors = {
    "raw": "C0",
    "pca": "C1",
    "fft": "C2",
    "fft_pca": "C3",
    "features": "C4",
    "features_pca": "C5",
}

method_labels = {
    "raw": "Raw",
    "pca": "Raw + PCA",
    "fft": "FFT",
    "fft_pca": "FFT + PCA",
    "features": "Catch22",
    "features_pca": "Catch22 + PCA",
}

kernel_styles = {
    "Linear": {"linestyle": "-",  "filled": True},
    "RBF":    {"linestyle": "--", "filled": False},
}


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

for kernel in ["Linear", "RBF"]:

    for method, color in method_colors.items():

        data = df_grouped[
            (df_grouped["Kernel"] == kernel) &
            (df_grouped["Method"] == method)
        ].sort_values("df")

        style = kernel_styles[kernel]

        plt.errorbar(
            data["df"],
            data["AUC_mean"],
            yerr=data["AUC_std"],

            color=color,
            linestyle='none',

            marker="o",
            #markersize=5,

            markerfacecolor=color if style["filled"] else "white",
            markeredgecolor=color,
            markeredgewidth=1.2,

            capsize=4,
            alpha=0.85,
        )


# -----------------------------
# Labels & layout
# -----------------------------
plt.xlabel(r"Parameter difference $(\nu - \nu_0)$")
plt.ylabel("AUC")
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.1)

plt.text(
    -0.13, 1.01,
    "(a)",
    fontweight="bold",
    fontsize=14,
    va="bottom",
    ha="left",
    transform=plt.gca().transAxes
)


plt.tight_layout()
plt.savefig('notebooks/linear_comparations_sine.eps', format='eps')
plt.show()

# -----------------------------
# Method legend (colors)
# -----------------------------
method_handles = [
    Line2D(
        [0], [0],
        marker='o',
        color=method_colors[m],
        linestyle='None',
        markersize=7,
        label=method_labels[m]
    )
    for m in method_colors
]


# -----------------------------
# Kernel legend (fill style)
# -----------------------------
kernel_handles = [
    Line2D(
        [0], [0],
        marker='o',
        color='black',
        linestyle='None',
        markerfacecolor='black',
        markeredgecolor='black',
        label='Linear'
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='black',
        linestyle='None',
        markerfacecolor='white',
        markeredgecolor='black',
        label='RBF'
    ),
]


# -----------------------------
# Create legend-only figure
# -----------------------------
# -----------------------------
# Method legend (colors)
# -----------------------------
method_handles = [
    Line2D(
        [0], [0],
        marker='s',
        color=method_colors[m],
        linestyle='None',
        #markersize=7,
        label=method_labels[m]
    )
    for m in method_colors
]


# -----------------------------
# Kernel legend (fill style)
# -----------------------------
kernel_handles = [
    Line2D(
        [0], [0],
        marker='s',
        color='black',
        linestyle='None',
        markerfacecolor='black',
        markeredgecolor='black',
        label='Linear'
    ),
    Line2D(
        [0], [0],
        marker='s',
        color='black',
        linestyle='None',
        markerfacecolor='white',
        markeredgecolor='black',
        label='RBF'
    ),
]


# -----------------------------
# Combine all legend entries
# -----------------------------
all_handles = method_handles + kernel_handles


# -----------------------------
# Figure (legend only)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 2.2))
ax.axis("off")

ax.legend(
    handles=all_handles,
    loc="center",
    ncol=4,              # <-- 4 COLUMNS HERE
    frameon=False,
    columnspacing=1.8,
    handletextpad=0.6
)

plt.tight_layout()
plt.savefig('notebooks/linear_comparations_legend_fhn_dyn.eps', format='eps')
plt.show()