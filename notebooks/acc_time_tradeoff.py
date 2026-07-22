import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("report.mplstyle")


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing the sinusoidal parameter-difference results
RESULTS_FOLDER = Path("results/sine/sine_frequency")

# Pattern identifying the correct pickle file
FILE_PATTERN = "results_20260610*.pkl"

# Check the name used inside each dictionary in all_results.
# Possible examples: "frequency", "df", "delta_f", "nu", or "b".
PARAMETER_KEY = "df"

# Select a frequency difference that does not give perfect classification.
# Replace this value with the condition you want to show.
TARGET_PARAMETER_DIFFERENCE = 0.18

# Output location
OUTPUT_FOLDER = RESULTS_FOLDER / "performance_time_tradeoff"

# Use logarithmic scale because Catch22 preprocessing is much slower
USE_LOG_SCALE = True

# Figure output resolution
PNG_DPI = 300


# ============================================================
# METHODS, COLORS, MARKERS, AND LABELS
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

method_markers = {
    "raw": "o",
    "pca": "s",
    "fft": "^",
    "fft_pca": "P",
    "features": "D",
    "features_pca": "X",
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
# SINUSOIDAL TIMING RESULTS FROM TABLE 3
#
# Values are:
# mean time, standard deviation
#
# Units: milliseconds
# ============================================================

timing_results = {
    "raw": {
        "pre": (0.56, 0.04),
        "train": (5.21, 0.73),
        "test": (0.44, 0.05),
    },
    "pca": {
        "pre": (1.52, 0.08),
        "train": (4.96, 0.58),
        "test": (0.35, 0.04),
    },
    "fft": {
        "pre": (0.64, 0.05),
        "train": (4.38, 0.77),
        "test": (0.34, 0.05),
    },
    "fft_pca": {
        "pre": (1.35, 0.07),
        "train": (4.16, 0.62),
        "test": (0.34, 0.05),
    },
    "features": {
        "pre": (31.59, 0.30),
        "train": (4.18, 0.82),
        "test": (0.40, 0.11),
    },
    "features_pca": {
        "pre": (33.04, 0.33),
        "train": (4.64, 0.60),
        "test": (0.48, 0.10),
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_result_file(folder, pattern):
    """
    Find the most recent pickle file matching the pattern.
    """
    matches = sorted(folder.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No pickle file matching '{pattern}' was found in:\n"
            f"{folder.resolve()}"
        )

    if len(matches) > 1:
        print("Several matching files were found:")
        for file in matches:
            print(f"  {file}")

        print(f"\nUsing the last matching file:\n{matches[-1]}")

    return matches[-1]


def inspect_results(all_results):
    """
    Print the available keys in the first result entry.
    """
    if len(all_results) == 0:
        raise ValueError("The pickle file contains no results.")

    print("\nKeys in the first entry:")
    print(list(all_results[0].keys()))


def convert_results_to_dataframe(
    all_results,
    parameter_key,
    methods,
):
    """
    Convert the nested pickle structure into a long-format DataFrame.
    """
    records = []

    for entry_index, entry in enumerate(all_results):

        if parameter_key not in entry:
            raise KeyError(
                f"Parameter key '{parameter_key}' was not found in "
                f"entry {entry_index}.\n"
                f"Available keys: {list(entry.keys())}"
            )

        parameter_value = entry[parameter_key]

        for method in methods:

            if method not in entry:
                raise KeyError(
                    f"Method '{method}' was not found in entry "
                    f"{entry_index}.\n"
                    f"Available keys: {list(entry.keys())}"
                )

            auc_values = np.asarray(
                entry[method],
                dtype=float,
            ).ravel()

            for auc in auc_values:
                records.append(
                    {
                        "Parameter_difference": float(parameter_value),
                        "Method": method,
                        "AUC": float(auc),
                    }
                )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No AUC values were extracted from the pickle file."
        )

    return df


def select_parameter_value(
    df,
    requested_value,
):
    """
    Select the requested parameter value.

    If the exact floating-point value is unavailable, the closest
    available value is selected.
    """
    available_values = np.sort(
        df["Parameter_difference"].unique()
    )

    close_match = available_values[
        np.isclose(
            available_values,
            requested_value,
            rtol=1e-7,
            atol=1e-10,
        )
    ]

    if len(close_match) > 0:
        selected_value = float(close_match[0])

    else:
        selected_value = float(
            available_values[
                np.argmin(
                    np.abs(
                        available_values - requested_value
                    )
                )
            ]
        )

        print(
            f"\nWarning: {requested_value} was not found exactly."
        )
        print(
            f"The closest available value will be used: "
            f"{selected_value}"
        )

    selected_df = df[
        np.isclose(
            df["Parameter_difference"],
            selected_value,
            rtol=1e-7,
            atol=1e-10,
        )
    ].copy()

    return selected_df, selected_value


def calculate_auc_statistics(
    selected_df,
    methods,
):
    """
    Compute mean, standard deviation, and sample count for AUC.
    """
    auc_stats = (
        selected_df
        .groupby("Method", as_index=False)
        .agg(
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
            AUC_count=("AUC", "count"),
        )
    )

    missing_methods = set(methods) - set(
        auc_stats["Method"]
    )

    if missing_methods:
        raise ValueError(
            f"Missing methods for the selected parameter value: "
            f"{sorted(missing_methods)}"
        )

    auc_stats["Method"] = pd.Categorical(
        auc_stats["Method"],
        categories=methods,
        ordered=True,
    )

    auc_stats = (
        auc_stats
        .sort_values("Method")
        .reset_index(drop=True)
    )

    return auc_stats


def combine_timing_statistics(
    auc_stats,
    timing_results,
):
    """
    Combine AUC values with timing values from Table 3.

    Standard deviations are propagated assuming independence:

    sigma_total = sqrt(sigma_pre^2 + sigma_other^2)
    """
    rows = []

    for _, auc_row in auc_stats.iterrows():

        method = str(auc_row["Method"])

        pre_mean, pre_std = timing_results[method]["pre"]
        train_mean, train_std = timing_results[method]["train"]
        test_mean, test_std = timing_results[method]["test"]

        pre_train_mean = pre_mean + train_mean

        pre_train_std = np.sqrt(
            pre_std**2 + train_std**2
        )

        pre_test_mean = pre_mean + test_mean

        pre_test_std = np.sqrt(
            pre_std**2 + test_std**2
        )

        rows.append(
            {
                "Method": method,
                "AUC_mean": auc_row["AUC_mean"],
                "AUC_std": auc_row["AUC_std"],
                "AUC_count": int(
                    auc_row["AUC_count"]
                ),
                "Preprocessing_mean_ms": pre_mean,
                "Preprocessing_std_ms": pre_std,
                "Training_mean_ms": train_mean,
                "Training_std_ms": train_std,
                "Testing_mean_ms": test_mean,
                "Testing_std_ms": test_std,
                "Preprocessing_training_mean_ms": (
                    pre_train_mean
                ),
                "Preprocessing_training_std_ms": (
                    pre_train_std
                ),
                "Preprocessing_testing_mean_ms": (
                    pre_test_mean
                ),
                "Preprocessing_testing_std_ms": (
                    pre_test_std
                ),
            }
        )

    return pd.DataFrame(rows)


def save_figure(
    fig,
    output_folder,
    filename_stem,
):
    """
    Save the figure as EPS, PDF, and PNG.
    """
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_folder / f"{filename_stem}.eps",
        format="eps",
        dpi=180,
        bbox_inches="tight",
    )
'''
    fig.savefig(
        output_folder / f"{filename_stem}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_folder / f"{filename_stem}.png",
        format="png",
        dpi=PNG_DPI,
        bbox_inches="tight",
    )
'''

def create_tradeoff_plot(
    plot_data,
    y_mean_column,
    y_std_column,
    y_label,
    panel_label,
    filename_stem,
    output_folder,
    use_log_scale=True,
):
    """
    Create an AUC versus computational time plot.

    The legend is not included inside this figure.
    It is exported separately.
    """
    fig, ax = plt.subplots(
        figsize=(6.4, 4.8)
    )

    for method in METHODS:

        row = plot_data[
            plot_data["Method"] == method
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        ax.errorbar(
            row["AUC_mean"],
            row[y_mean_column],
            xerr=row["AUC_std"],
            yerr=row[y_std_column],
            fmt=method_markers[method],
            color=method_colors[method],
            markeredgecolor=method_colors[method],
            markersize=9,
            capsize=5,
            elinewidth=1.2,
            alpha=0.85,
            label=method_labels[method],
        )

    if use_log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("AUC")
    ax.set_ylabel(y_label)

    ax.grid(
        True,
        which="both",
        alpha=0.35,
    )

    # Adjust this after looking at your data if necessary
    ax.set_xlim(0.3, 1.1)
    ax.set_ylim(0.1, 100)

    ax.text(
        -0.13,
        1.01,
        panel_label,
        fontweight="bold",
        fontsize=14,
        va="bottom",
        ha="left",
        transform=ax.transAxes,
    )

    fig.tight_layout()

    save_figure(
        fig=fig,
        output_folder=output_folder,
        filename_stem=filename_stem,
    )

    return fig, ax


def create_separate_legend(
    output_folder,
):
    """
    Create and save one separate horizontal legend.
    """
    fig = plt.figure(
        figsize=(8.5, 1.0)
    )

    handles = []

    for method in METHODS:

        handle = plt.Line2D(
            [],
            [],
            color=method_colors[method],
            marker=method_markers[method],
            linestyle="None",
            markersize=9,
            label=method_labels[method],
        )

        handles.append(handle)

    legend = fig.legend(
        handles=handles,
        labels=[
            method_labels[method]
            for method in METHODS
        ],
        ncol=6,
        loc="center",
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    fig.canvas.draw()

    bbox = (
        legend
        .get_window_extent()
        .transformed(
            fig.dpi_scale_trans.inverted()
        )
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_folder / "tradeoff_legend.eps",
        format="eps",
        dpi=180,
        bbox_inches=bbox,
    )

    '''fig.savefig(
        output_folder / "tradeoff_legend.pdf",
        format="pdf",
        bbox_inches=bbox,
    )

    fig.savefig(
        output_folder / "tradeoff_legend.png",
        format="png",
        dpi=PNG_DPI,
        bbox_inches=bbox,
    )'''

    return fig


# ============================================================
# LOAD PICKLE FILE
# ============================================================

result_file = find_result_file(
    folder=RESULTS_FOLDER,
    pattern=FILE_PATTERN,
)

print(
    f"\nLoading results from:\n"
    f"{result_file.resolve()}"
)

with open(result_file, "rb") as file:
    all_results = pickle.load(file)

inspect_results(all_results)


# ============================================================
# CONVERT PICKLE RESULTS TO DATAFRAME
# ============================================================

df_results = convert_results_to_dataframe(
    all_results=all_results,
    parameter_key=PARAMETER_KEY,
    methods=METHODS,
)

print("\nAvailable parameter differences:")

print(
    np.sort(
        df_results[
            "Parameter_difference"
        ].unique()
    )
)


# ============================================================
# SELECT THE DESIRED PARAMETER DIFFERENCE
# ============================================================

df_selected, selected_parameter = (
    select_parameter_value(
        df=df_results,
        requested_value=(
            TARGET_PARAMETER_DIFFERENCE
        ),
    )
)

print(
    f"\nSelected parameter difference: "
    f"{selected_parameter}"
)


# ============================================================
# CALCULATE AUC STATISTICS
# ============================================================

auc_stats = calculate_auc_statistics(
    selected_df=df_selected,
    methods=METHODS,
)


# ============================================================
# COMBINE AUC AND TIMING STATISTICS
# ============================================================

plot_data = combine_timing_statistics(
    auc_stats=auc_stats,
    timing_results=timing_results,
)

print("\nValues used in the figures:")

print(
    plot_data[
        [
            "Method",
            "AUC_mean",
            "AUC_std",
            "AUC_count",
            "Preprocessing_training_mean_ms",
            "Preprocessing_training_std_ms",
            "Preprocessing_testing_mean_ms",
            "Preprocessing_testing_std_ms",
        ]
    ].to_string(index=False)
)


# ============================================================
# CREATE OUTPUT FOLDER AND EXPORT DATA
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

safe_parameter_string = (
    str(selected_parameter)
    .replace(".", "p")
)

csv_file = (
    OUTPUT_FOLDER
    / (
        "auc_time_tradeoff_"
        f"delta_nu_{safe_parameter_string}.csv"
    )
)

plot_data.to_csv(
    csv_file,
    index=False,
)

print(
    f"\nPlot data saved to:\n"
    f"{csv_file.resolve()}"
)


# ============================================================
# FIGURE A
# AUC VS PREPROCESSING + TRAINING TIME
# ============================================================

fig_train, ax_train = create_tradeoff_plot(
    plot_data=plot_data,
    y_mean_column=(
        "Preprocessing_training_mean_ms"
    ),
    y_std_column=(
        "Preprocessing_training_std_ms"
    ),
    y_label=(
        "Preprocessing + training time (ms)"
    ),
    panel_label="(a)",
    filename_stem=(
        "auc_vs_preprocessing_training_"
        f"delta_nu_{safe_parameter_string}"
    ),
    output_folder=OUTPUT_FOLDER,
    use_log_scale=USE_LOG_SCALE,
)


# ============================================================
# FIGURE B
# AUC VS PREPROCESSING + TESTING TIME
# ============================================================

fig_test, ax_test = create_tradeoff_plot(
    plot_data=plot_data,
    y_mean_column=(
        "Preprocessing_testing_mean_ms"
    ),
    y_std_column=(
        "Preprocessing_testing_std_ms"
    ),
    y_label=(
        "Preprocessing + testing time (ms)"
    ),
    panel_label="(b)",
    filename_stem=(
        "auc_vs_preprocessing_testing_"
        f"delta_nu_{safe_parameter_string}"
    ),
    output_folder=OUTPUT_FOLDER,
    use_log_scale=USE_LOG_SCALE,
)


# ============================================================
# SEPARATE LEGEND
# ============================================================

legend_fig = create_separate_legend(
    output_folder=OUTPUT_FOLDER,
)


# ============================================================
# DISPLAY FIGURES
# ============================================================

plt.show()