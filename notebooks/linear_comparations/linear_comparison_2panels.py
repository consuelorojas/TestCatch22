import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("report.mplstyle")


# ============================================================
# CONFIGURATION
# ============================================================

SIGNAL_NAME = "fhn_dyn"

linear_file = Path(
    "results/fhn/fhn_parameter/"
    "results_lineal_20260701_124831.pkl"
)

rbf_file = Path(
    "results/fhn/fhn_parameter/"
    "results_20260616_122741.pkl"
)

OUTPUT_FIGURE = Path(
    "notebooks/linear_rbf_comparison_fhn_dyn.eps"
)
    
OUTPUT_LEGEND = Path(
    "notebooks/linear_rbf_comparison_legend.eps"
)


# ============================================================
# GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
})


# ============================================================
# METHODS
# ============================================================

METHODS = [
    "raw",
    "pca",
    "fft",
    "fft_pca",
    "features",
    "features_pca",
]

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


# ============================================================
# MARKERS REPRESENT SIGNAL TYPE
# ============================================================

markers = {
    "sine": "o",
    "fhn_obs": "^",
    "fhn_dyn": "s",
}

signal_labels = {
    "sine": "Sinusoidal",
    "fhn_obs": "FHN observational",
    "fhn_dyn": "FHN dynamic",
}

if SIGNAL_NAME not in markers:
    raise ValueError(
        f"Unknown signal name '{SIGNAL_NAME}'. "
        f"Choose from: {list(markers.keys())}"
    )

signal_marker = markers[SIGNAL_NAME]


# ============================================================
# LOAD RESULTS
# ============================================================

if not linear_file.exists():
    raise FileNotFoundError(
        f"Linear-results file not found:\n"
        f"{linear_file.resolve()}"
    )

if not rbf_file.exists():
    raise FileNotFoundError(
        f"RBF-results file not found:\n"
        f"{rbf_file.resolve()}"
    )

with open(linear_file, "rb") as file:
    linear_results = pickle.load(file)

with open(rbf_file, "rb") as file:
    rbf_results = pickle.load(file)


# ============================================================
# CONVERT RESULTS TO DATAFRAME
# ============================================================

def results_to_dataframe(results, kernel):
    """
    Convert the nested result structure into a long-format DataFrame.

    Expected structure:
        entry["df"]   -> parameter difference
        entry[method] -> repeated AUC values
    """
    records = []

    for entry_index, entry in enumerate(results):

        if "b" not in entry:
            raise KeyError(
                f"The key 'df' was not found in entry {entry_index}. "
                f"Available keys: {list(entry.keys())}"
            )

        df_value = entry["b"]

        for method in METHODS:

            if method not in entry:
                raise KeyError(
                    f"Method '{method}' was not found in entry "
                    f"{entry_index}. Available keys: "
                    f"{list(entry.keys())}"
                )

            for auc in entry[method]:
                records.append({
                    "df": float(df_value),
                    "Method": method,
                    "Kernel": kernel,
                    "AUC": float(auc),
                })

    return pd.DataFrame(records)


df_results = pd.concat(
    [
        results_to_dataframe(
            linear_results,
            "Linear",
        ),
        results_to_dataframe(
            rbf_results,
            "RBF",
        ),
    ],
    ignore_index=True,
)


# ============================================================
# AGGREGATE STATISTICS
# ============================================================

df_grouped = (
    df_results
    .groupby(
        ["Kernel", "Method", "df"],
        as_index=False,
    )
    .agg(
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
    )
)


# ============================================================
# PLOT ONE KERNEL PANEL
# ============================================================

def plot_kernel_panel(
    ax,
    grouped_data,
    kernel,
    panel_label,
):
    """
    Plot one classifier kernel in one panel.

    Color identifies the representation.
    Marker shape identifies the signal type.
    """
    for method in METHODS:

        data = grouped_data[
            (grouped_data["Kernel"] == kernel)
            & (grouped_data["Method"] == method)
        ].sort_values("df")

        if data.empty:
            continue

        ax.errorbar(
            data["df"],
            data["AUC_mean"],
            yerr=data["AUC_std"],

            color=method_colors[method],
            linestyle="none",

            # Marker represents signal type
            marker=signal_marker,
            markerfacecolor=method_colors[method],
            markeredgecolor=method_colors[method],
            markeredgewidth=1.2,
            markersize=8,

            capsize=4,
            elinewidth=1.1,
            alpha=0.85,
        )

    ax.set_title(
        f"{kernel} SVM",
        fontsize=21,
        pad=10,
    )

    ax.set_xlabel(
        r"Parameter difference $(b-b_0)$",
        fontsize=20,
    )

    ax.set_ylim(0.0, 1.1)

    ax.grid(
        True,
        alpha=0.3,
    )

    '''
    ax.text(
        0.02,
        0.96,
        f"({panel_label})",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="left",
    )
    '''

    ax.tick_params(
        axis="both",
        labelsize=16,
    )


# ============================================================
# CREATE TWO-COLUMN FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(13.5, 5.8),
    sharex=True,
    sharey=True,
)

plot_kernel_panel(
    ax=axes[0],
    grouped_data=df_grouped,
    kernel="Linear",
    panel_label="a",
)

plot_kernel_panel(
    ax=axes[1],
    grouped_data=df_grouped,
    kernel="RBF",
    panel_label="b",
)

axes[0].set_ylabel(
    "AUC",
    fontsize=20,
)

fig.tight_layout()

OUTPUT_FIGURE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT_FIGURE,
    dpi=600,
    bbox_inches="tight",
)

print(f"Saved figure:\n{OUTPUT_FIGURE.resolve()}")


# ============================================================
# SEPARATE METHOD LEGEND
#
# Colors represent methods.
# All markers use the signal-specific marker.
# ============================================================

method_handles = [
    Line2D(
        [0],
        [0],
        marker=signal_marker,
        color=method_colors[method],
        markerfacecolor=method_colors[method],
        markeredgecolor=method_colors[method],
        linestyle="none",
        markersize=9,
        label=method_labels[method],
    )
    for method in METHODS
]

legend_fig = plt.figure(
    figsize=(13.5, 1.2)
)

legend = legend_fig.legend(
    handles=method_handles,
    loc="center",
    ncol=6,
    frameon=False,
    fontsize=17,
    handletextpad=0.5,
    columnspacing=1.5,
)

legend_fig.canvas.draw()

legend_bbox = (
    legend
    .get_window_extent()
    .transformed(
        legend_fig.dpi_scale_trans.inverted()
    )
)

legend_fig.savefig(
    OUTPUT_LEGEND,
    dpi=600,
    bbox_inches=legend_bbox,
)

print(f"Saved legend:\n{OUTPUT_LEGEND.resolve()}")

plt.show()