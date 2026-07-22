import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.style.use("report.mplstyle")


# ============================================================
# GLOBAL FONT SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 25,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
})


# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    "sine": "results/sine/sine_periods/results_times_20260701_124139.pkl",
    "fhn_obs": "results/fhn_obs/periods/results_times_20260701_124139.pkl",
    "fhn_dyn": "results/fhn/fhn_periods/results_times_20260701_124139.pkl",
}

METHODS = [
    "raw",
    "fft",
    "features",
]

COLORS = {
    "raw": "C0",
    "fft": "C2",
    "features": "C4",
}

METHOD_LABELS = {
    "raw": "Raw",
    "fft": "FFT",
    "features": "Catch22",
}

PANEL_LABELS = {
    "sine": "a",
    "fhn_obs": "b",
    "fhn_dyn": "c",
}

OUTPUT_DPI = 600


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def mean_times(fold_list):
    """
    Calculate the mean computational times over all folds.

    Parameters
    ----------
    fold_list : list
        List of tuples with the structure:

        (training_time, testing_time, preprocessing_time, tuning_time)

        Times are assumed to be stored in seconds.

    Returns
    -------
    dict
        Mean times converted to milliseconds.
    """
    arr = np.asarray(fold_list, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(
            "Each method entry must contain rows with the structure "
            "(train, test, pre, tune)."
        )

    return {
        "train": arr[:, 0].mean() * 1000,
        "test": arr[:, 1].mean() * 1000,
        "pre": arr[:, 2].mean() * 1000,
        "tune": arr[:, 3].mean() * 1000,
    }


def extract_method_over_sweep(data, method):
    """
    Extract mean timing statistics for one method across all sweep points.

    Parameters
    ----------
    data : list
        Complete data loaded from one pickle file.

    method : str
        Method name, such as 'raw', 'fft', or 'features'.

    Returns
    -------
    list of dict
        One timing dictionary per sweep point.
    """
    output = []

    for index, sweep_point in enumerate(data):

        if method not in sweep_point:
            raise KeyError(
                f"Method '{method}' was not found at sweep point {index}. "
                f"Available keys: {list(sweep_point.keys())}"
            )

        method_data = sweep_point[method]
        output.append(mean_times(method_data))

    return output


def validate_periods(data):
    """
    Verify that every sweep point contains the 'periods' key.
    """
    missing = [
        index
        for index, sweep_point in enumerate(data)
        if "periods" not in sweep_point
    ]

    if missing:
        raise KeyError(
            "The key 'periods' is missing from sweep points: "
            f"{missing}"
        )


def save_figure(fig, filename_stem):
    """
    Save a figure as PDF, EPS, and PNG.
    """
    fig.savefig(
        f"{filename_stem}.pdf",
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        f"{filename_stem}.eps",
        format="eps",
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        f"{filename_stem}.png",
        format="png",
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )


# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_signal(data, signal_name, panel_label):
    """
    Create one stacked grouped-bar plot for a signal type.

    Each group corresponds to one number of periods.
    Within each group, the three bars correspond to:
        Raw, FFT, and Catch22.

    Each bar is divided into:
        preprocessing, training, testing, and tuning.
    """
    validate_periods(data)

    samples = [sweep_point["periods"] for sweep_point in data]

    processed = {
        method: extract_method_over_sweep(data, method)
        for method in METHODS
    }

    x = np.arange(len(samples))

    bar_width = 0.28

    offsets = {
        "raw": -bar_width,
        "fft": 0,
        "features": bar_width,
    }

    fig, ax = plt.subplots(figsize=(15, 7))

    for method in METHODS:

        pre = np.array([
            processed[method][j]["pre"]
            for j in range(len(samples))
        ])

        train = np.array([
            processed[method][j]["train"]
            for j in range(len(samples))
        ])

        test = np.array([
            processed[method][j]["test"]
            for j in range(len(samples))
        ])

        tune = np.array([
            processed[method][j]["tune"]
            for j in range(len(samples))
        ])

        x_position = x + offsets[method]

        bottom = np.zeros(len(samples))

        # Preprocessing
        ax.bar(
            x_position,
            pre,
            bar_width,
            color=COLORS[method],
            alpha=0.90,
            edgecolor="none",
        )

        bottom += pre

        # Training
        ax.bar(
            x_position,
            train,
            bar_width,
            bottom=bottom,
            color=COLORS[method],
            alpha=0.65,
            edgecolor="none",
        )

        bottom += train

        # Testing
        ax.bar(
            x_position,
            test,
            bar_width,
            bottom=bottom,
            color=COLORS[method],
            alpha=0.40,
            edgecolor="none",
        )

        bottom += test

        # Hyperparameter tuning
        ax.bar(
            x_position,
            tune,
            bar_width,
            bottom=bottom,
            color=COLORS[method],
            alpha=0.20,
            edgecolor="none",
        )

    # Panel label
    ax.text(
        0.02,
        0.96,
        f"({panel_label})",
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        va="top",
        ha="left",
    )

    # Axis labels
    ax.set_xlabel(
        r"Number of periods ($N_p$)",
        fontsize=24,
        labelpad=12,
    )

    ax.set_ylabel(
        "Time (ms)",
        fontsize=24,
        labelpad=12,
    )

    # Tick settings
    ax.set_xticks(x)

    ax.set_xticklabels(
        samples,
        rotation=45,
        ha="right",
        fontsize=19,
    )

    ax.tick_params(
        axis="y",
        labelsize=19,
    )

    ax.tick_params(
        axis="both",
        width=1.4,
        length=6,
    )

    # Grid
    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    fig.tight_layout()

    return fig, ax


# ============================================================
# SEPARATE ONE-ROW LEGEND
# ============================================================

def build_legend():
    """
    Create one independent legend figure with all entries in one row.
    """

    method_handles = [
        Patch(
            facecolor=COLORS["raw"],
            edgecolor="none",
            label="Raw",
        ),
        Patch(
            facecolor=COLORS["fft"],
            edgecolor="none",
            label="FFT",
        ),
        Patch(
            facecolor=COLORS["features"],
            edgecolor="none",
            label="Catch22",
        ),
    ]

    time_handles = [
        Patch(
            facecolor="black",
            edgecolor="none",
            alpha=0.90,
            label="Preprocessing",
        ),
        Patch(
            facecolor="black",
            edgecolor="none",
            alpha=0.65,
            label="Training",
        ),
        Patch(
            facecolor="black",
            edgecolor="none",
            alpha=0.40,
            label="Testing",
        ),
        Patch(
            facecolor="black",
            edgecolor="none",
            alpha=0.20,
            label="Tuning",
        ),
    ]

    all_handles = method_handles + time_handles

    fig = plt.figure(figsize=(17, 1.5))

    legend = fig.legend(
        handles=all_handles,
        loc="center",
        ncol=7,
        frameon=False,
        fontsize=20,
        handlelength=1.8,
        handleheight=1.2,
        columnspacing=1.5,
        handletextpad=0.6,
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

    for signal_name, file_path in FILES.items():

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Timing file not found:\n{file_path.resolve()}"
            )

        print(f"Loading: {file_path}")

        with open(file_path, "rb") as file:
            data = pickle.load(file)

        fig, ax = plot_signal(
            data=data,
            signal_name=signal_name,
            panel_label=PANEL_LABELS[signal_name],
        )

        save_figure(
            fig=fig,
            filename_stem=f"{signal_name}_timing_periods",
        )

    # --------------------------------------------------------
    # SEPARATE GLOBAL LEGEND
    # --------------------------------------------------------

    legend_fig, legend_bbox = build_legend()

    legend_fig.savefig(
        "legend_times.pdf",
        dpi=OUTPUT_DPI,
        bbox_inches=legend_bbox,
    )

    legend_fig.savefig(
        "legend_times.eps",
        format="eps",
        dpi=OUTPUT_DPI,
        bbox_inches=legend_bbox,
    )

    legend_fig.savefig(
        "legend_times.png",
        format="png",
        dpi=OUTPUT_DPI,
        bbox_inches=legend_bbox,
    )

    plt.show()


if __name__ == "__main__":
    main()