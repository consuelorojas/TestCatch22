import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("report.mplstyle")

# --------------------------------------------------
# CONFIG (EDIT ONLY THIS PER FIGURE)
# --------------------------------------------------

CONFIGS = {
    "noise": {
        "files": {
            "fhn_obs": "results/fhn_obs/noise",
            "fhn_dyn": "results/fhn/fhn_noise",
        },
        "pattern": "results_202606*.pkl",
        "x": "noise",
        "xlabel": r"Noise strength $(D_{obs})$",
        "xlim": None,
        "ylim": (0.5, 1.05),
        "label": "(a)",
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
        "label": "(b)",
    },

    "npp": {
        "files": {
            "fhn_obs": "results/fhn_obs/npoints",
            "fhn_dyn": "results/fhn/fhn_npp",
        },
        "pattern": "results_202606*.pkl",
        "x": "npp",
        "xlabel": r"Number of points per period $(N_{pp})$",
        "xlim": None,
        "ylim": (0.4, 1.05),
        "label": "(c)",
    },

    "samples": {
        "files": {
            "fhn_obs": "results/fhn_obs/samples",
            "fhn_dyn": "results/fhn/fhn_samples",
        },
        "pattern": "results_202606*.pkl",
        "x": "samples",
        "xlabel": r"Samples $(N_s)$",
        "xlim": (0, 255),
        "ylim": (0.0, 1.1),
        "label": "(e)",
    },

    "b": {
        "files": {
            "fhn_obs": "results/fhn_obs/parameter",
            "fhn_dyn": "results/fhn/fhn_parameter",
        },
        "pattern": "results_202606*.pkl",
        "x": "b",
        "xlabel": r"Parameter difference $(b - b_0)$",
        "xlim": (-0.01, 0.31),
        "ylim": (0.2, 1.05),
        "label": "(d)",
    },
}

# --------------------------------------------------
# STYLE (GLOBAL)
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

methods = list(method_colors.keys())

# --------------------------------------------------
# CORE FUNCTION
# --------------------------------------------------

def plot_experiment(param_key):
    cfg = CONFIGS[param_key]

    records = []

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    for signal, folder in cfg["files"].items():

        result_file = next(Path(folder).glob(cfg["pattern"]))

        with open(result_file, "rb") as f:
            all_results = pickle.load(f)

        for entry in all_results:

            xval = entry[cfg["x"]]

            for method in methods:
                for auc in entry[method]:

                    records.append(
                        {
                            "Signal": signal,
                            cfg["x"]: xval,
                            "Method": method,
                            "AUC": auc,
                        }
                    )

    df = pd.DataFrame(records)
    df[cfg["x"]] = df[cfg["x"]].astype(float).round(4)

    # -----------------------------
    # GROUP
    # -----------------------------
    df_grouped = (
        df
        .groupby(["Signal", "Method", cfg["x"]])
        .agg(
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
        )
        .reset_index()
    )

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure(figsize=(6.4, 4.8))

    for method, color in method_colors.items():

        for signal, marker in signal_markers.items():

            data = (
                df_grouped[
                    (df_grouped["Method"] == method)
                    & (df_grouped["Signal"] == signal)
                ]
                .sort_values(cfg["x"])
            )

            plt.errorbar(
                data[cfg["x"]],
                data["AUC_mean"],
                yerr=data["AUC_std"],
                fmt=marker,
                color=color,
                capsize=5,
                linestyle="None",
            )
    # -----------------------------
    # AXES
    # -----------------------------
    plt.xlabel(cfg["xlabel"])
    plt.ylabel("AUC")

    plt.grid(True, alpha=0.4)

    plt.ylim(*cfg["ylim"])

    if cfg["xlim"] is not None:
        plt.xlim(*cfg["xlim"])

    # clean xticks
    xticks = sorted(df_grouped[cfg["x"]].unique())
    xticks = xticks[::2]
    plt.xticks(xticks)

    # panel label
    plt.text(
        -0.13,
        1.01,
        cfg["label"],
        fontweight="bold",
        fontsize=13,
        va="bottom",
        ha="left",
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()

    # -----------------------------
    # SAVE
    # -----------------------------
    plt.savefig(
        f"fhn_{param_key}_combined.eps",
        format="eps",
        dpi=180,
    )

    plt.show()

plot_experiment("noise")
plot_experiment("periods")
plot_experiment("npp")
plot_experiment("samples")
plot_experiment("b")