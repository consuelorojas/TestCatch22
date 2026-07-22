import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.style.use("report.mplstyle")


# ============================================================
# GLOBAL SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
})


# ============================================================
# EXPERIMENT CONFIGURATION
#
# The order here determines the row order in the final figure.
# ============================================================

CONFIGS = {
    "b": {
        "files": {
            "fhn_obs": "results/fhn_obs/parameter",
            "fhn_dyn": "results/fhn/fhn_parameter",
        },
        "pattern": "results_202606*.pkl",
        "x": "b",
        "xlabel": r"Parameter difference $(b-b_0)$",
        "xlim": {
            "fhn_obs": (-0.005, 0.105),
            "fhn_dyn": (-0.01, 0.31),
        },
        "ylim": (0.2, 1.05),
    },

    "periods": {
        "files": {
            "fhn_obs": "results/fhn_obs/periods",
            "fhn_dyn": "results/fhn/fhn_periods",
        },
        "pattern": "results_202606*.pkl",
        "x": "periods",
        "xlabel": r"Number of periods $(N_p)$",
        "xlim": None,
        "ylim": (0.2, 1.05),
    },

    "npp": {
        "files": {
            "fhn_obs": "results/fhn_obs/npoints",
            "fhn_dyn": "results/fhn/fhn_npp",
        },
        "pattern": "results_202606*.pkl",
        "x": "npp",
        "xlabel": r"Points per period $(N_{pp})$",
        "xlim": None,
        "ylim": (0.2, 1.05),
    },

    "noise": {
        "files": {
            "fhn_obs": "results/fhn_obs/noise",
            "fhn_dyn": "results/fhn/fhn_noise",
        },
        "pattern": "results_202606*.pkl",
        "x": "noise",
        "xlabel": r"Noise strength $(D)$",
        "xlim": None,
        "ylim": (0.2, 1.05),
    },

    "samples": {
        "files": {
            "fhn_obs": "results/fhn_obs/samples",
            "fhn_dyn": "results/fhn/fhn_samples",
        },
        "pattern": "results_202606*.pkl",
        "x": "samples",
        "xlabel": r"Number of samples $(N_s)$",
        "xlim": (0, 255),
        "ylim": (0.0, 1.05),
    },
}


# ============================================================
# METHOD STYLE
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


# Because signal type is now represented by the column,
# all points may use the same marker.
POINT_MARKER = "o"


# ============================================================
# SIGNAL SETTINGS
# ============================================================

SIGNALS = [
    "fhn_obs",
    "fhn_dyn",
]

signal_titles = {
    "fhn_obs": "Observational noise",
    "fhn_dyn": "Dynamic noise",
}


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = Path("fhn_observational_dynamic_comparison.eps")
OUTPUT_EPS = Path("fhn_observational_dynamic_comparison.eps")
OUTPUT_LEGEND = Path("fhn_observational_dynamic_legend.eps")

OUTPUT_DPI = 600


# ============================================================
# FILE LOADING
# ============================================================

def find_result_file(folder, pattern):
    """
    Find the most recent file matching the requested pattern.
    """
    folder = Path(folder)
    matches = sorted(folder.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' was found in:\n"
            f"{folder.resolve()}"
        )

    if len(matches) > 1:
        print(f"\nMultiple files found in {folder}:")
        for match in matches:
            print(f"  {match}")

        print(f"Using the last matching file: {matches[-1]}")

    return matches[-1]


def load_experiment_dataframe(config):
    """
    Load observational- and dynamic-noise results for one parameter.

    Returns a long-format DataFrame with:
        Signal, x value, Method, AUC
    """
    records = []
    x_key = config["x"]

    for signal, folder in config["files"].items():

        result_file = find_result_file(
            folder=folder,
            pattern=config["pattern"],
        )

        print(f"Loading {signal}: {result_file}")

        with open(result_file, "rb") as file:
            all_results = pickle.load(file)

        for entry_index, entry in enumerate(all_results):

            if x_key not in entry:
                raise KeyError(
                    f"Key '{x_key}' not found in entry {entry_index} "
                    f"of {result_file}.\n"
                    f"Available keys: {list(entry.keys())}"
                )

            x_value = entry[x_key]

            for method in METHODS:

                if method not in entry:
                    raise KeyError(
                        f"Method '{method}' not found in entry "
                        f"{entry_index} of {result_file}."
                    )

                auc_values = np.asarray(
                    entry[method],
                    dtype=float,
                ).ravel()

                for auc in auc_values:
                    records.append({
                        "Signal": signal,
                        "x": float(x_value),
                        "Method": method,
                        "AUC": float(auc),
                    })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"No results were extracted for parameter '{x_key}'."
        )

    # Reduce floating-point grouping problems
    df["x"] = df["x"].round(6)

    return df


def calculate_statistics(df):
    """
    Calculate mean AUC and standard deviation.
    """
    return (
        df
        .groupby(
            ["Signal", "Method", "x"],
            as_index=False,
        )
        .agg(
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
        )
    )


# ============================================================
# TICK REDUCTION
# ============================================================

def select_xticks(values, maximum_ticks=6):
    """
    Return a reduced set of x ticks so small panels remain readable.
    """
    values = np.sort(np.unique(values))

    if len(values) <= maximum_ticks:
        return values

    indices = np.linspace(
        0,
        len(values) - 1,
        maximum_ticks,
        dtype=int,
    )

    return values[indices]


# ============================================================
# PLOT ONE PANEL
# ============================================================

def plot_panel(
    ax,
    grouped_data,
    signal,
    config,
    panel_label,
    show_ylabel,
):
    """
    Plot one parameter and one noise type.
    """
    signal_data = grouped_data[
        grouped_data["Signal"] == signal
    ]

    for method in METHODS:

        method_data = (
            signal_data[
                signal_data["Method"] == method
            ]
            .sort_values("x")
        )

        if method_data.empty:
            continue

        ax.errorbar(
            method_data["x"],
            method_data["AUC_mean"],
            yerr=method_data["AUC_std"],
            fmt=POINT_MARKER,
            linestyle="none",
            color=method_colors[method],
            markerfacecolor=method_colors[method],
            markeredgecolor=method_colors[method],
            markeredgewidth=0.7,
            markersize=3.7,
            capsize=2.2,
            capthick=0.7,
            elinewidth=0.7,
            alpha=0.85,
        )

    ax.set_xlabel(
        config["xlabel"],
        labelpad=2,
    )

    if show_ylabel:
        ax.set_ylabel(
            "AUC",
            labelpad=3,
        )

    ax.set_ylim(*config["ylim"])

    xlim = config["xlim"]

    if isinstance(xlim, dict):
        signal_xlim = xlim.get(signal)

        if signal_xlim is not None:
            ax.set_xlim(*signal_xlim)

    elif xlim is not None:
        ax.set_xlim(*xlim)

    xticks = select_xticks(
        signal_data["x"].unique(),
        maximum_ticks=6,
    )

    ax.set_xticks(xticks)

    # Use compact tick formatting
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=8,
        length=3,
        pad=2,
    )

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.6,
    )

    ax.set_axisbelow(True)

    # Panel label inside the upper-left corner
    ax.text(
        -0.16,
        1.04,
        f"({panel_label})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )

# ============================================================
# CREATE 5 × 2 FIGURE
# ============================================================

def create_figure():

    parameter_order = [
        "b",
        "periods",
        "npp",
        "noise",
        "samples",
    ]

    panel_letters = [
        ["a", "b"],
        ["c", "d"],
        ["e", "f"],
        ["g", "h"],
        ["i", "j"],
    ]

    # Compact dimensions appropriate for five rows
    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(7.2, 12.0),
        sharey="row",
    )

    for row_index, parameter_key in enumerate(parameter_order):

        config = CONFIGS[parameter_key]

        raw_df = load_experiment_dataframe(config)
        grouped_df = calculate_statistics(raw_df)

        for column_index, signal in enumerate(SIGNALS):

            ax = axes[row_index, column_index]

            plot_panel(
                ax=ax,
                grouped_data=grouped_df,
                signal=signal,
                config=config,
                panel_label=panel_letters[row_index][column_index],
                show_ylabel=(column_index == 0),
            )

    # Column headings
    axes[0, 0].set_title(
        signal_titles["fhn_obs"],
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    axes[0, 1].set_title(
        signal_titles["fhn_dyn"],
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    # Reduce spacing while preserving labels
    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.06,
        top=0.93,
        hspace=0.55,
        wspace=0.16,
    )

    return fig, axes


# ============================================================
# LEGEND
# ============================================================

def create_method_legend():
    """
    Create a separate horizontal method legend.
    """
    handles = [
        Line2D(
            [0],
            [0],
            marker=POINT_MARKER,
            color=method_colors[method],
            markerfacecolor=method_colors[method],
            markeredgecolor=method_colors[method],
            linestyle="none",
            markersize=6,
            label=method_labels[method],
        )
        for method in METHODS
    ]

    fig = plt.figure(
        figsize=(7.2, 0.65)
    )

    legend = fig.legend(
        handles=handles,
        loc="center",
        ncol=6,
        frameon=False,
        fontsize=9,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0,
    )

    fig.canvas.draw()

    bbox = legend.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )

    return fig, bbox


# ============================================================
# MAIN
# ============================================================

def main():

    figure, axes = create_figure()

    figure.savefig(
        OUTPUT_FILE,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        OUTPUT_EPS,
        format="eps",
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    print(f"\nSaved PDF:\n{OUTPUT_FILE.resolve()}")
    print(f"\nSaved EPS:\n{OUTPUT_EPS.resolve()}")

    legend_figure, legend_bbox = create_method_legend()

    legend_figure.savefig(
        OUTPUT_LEGEND,
        dpi=OUTPUT_DPI,
        bbox_inches=legend_bbox,
    )

    print(f"\nSaved legend:\n{OUTPUT_LEGEND.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()