import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_fhn = "results/fhn/fhn_periods/results_times_20251125_185555.pkl"
results_fhn_obs = "results/fhn_obs/periods/results_times_20251125_185424.pkl"
results_sine = "results/sine/sine_periods/results_times_20251126_130959.pkl"

def export_legend(legend, filename="legend_times.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


# ---- Convert to DataFrame ----
'''
Data structure in all_results:
    dictionary with keys: "samples", "raw", "pca", "features", "features_pca".
    Each key (except "samples") contains a list of lists of times [train, test, pre] for each fold.
    There are 50 folds, so 50 entries per method.
'''

def files_to_dataframe(results_files):
    """
    Convert sweep results from a pickle file to a pandas DataFrame. For time_experiment results.1
    """
    with open(results_files, 'rb') as f:
        all_results = pickle.load(f)

    records = []
    for entry in all_results:
        df =  entry["periods"]
        for method in ["raw", "pca", "features", "features_pca"]:
            for times in entry[method]:
                records.append({"periods": df, "Method": method, "Train": times[0], "Test": times[1], "Pre": times[2]})

    df_results = pd.DataFrame(records)
    return df_results

sine_results = files_to_dataframe(results_sine)
fhn_dyn_results = files_to_dataframe(result_fhn)
fhn_obs_results = files_to_dataframe(results_fhn_obs)


# --- Compute mean & std per method/Δf ---
df_sine = (
    sine_results
    .groupby(["Method", "periods"])
    .agg(train_mean=("Train", "mean"), train_std=("Train", "std"),
         test_mean=("Test", "mean"), test_std=("Test", "std"),
         pre_mean=("Pre", "mean"), pre_std=("Pre", "std"))
    .reset_index()
)
df_fhn_dyn = (
    fhn_dyn_results
    .groupby(["Method", "periods"])
    .agg(train_mean=("Train", "mean"), train_std=("Train", "std"),
         test_mean=("Test", "mean"), test_std=("Test", "std"),
         pre_mean=("Pre", "mean"), pre_std=("Pre", "std")) 
    .reset_index()
)
df_fhn_obs = (
    fhn_obs_results
    .groupby(["Method", "periods"])
    .agg(train_mean=("Train", "mean"), train_std=("Train", "std"),
         test_mean=("Test", "mean"), test_std=("Test", "std"),
         pre_mean=("Pre", "mean"), pre_std=("Pre", "std")) 
    .reset_index()
)

# ---- Plot Config ----
signal_frames = {
    "Sine": df_sine,
    "FHN obs": df_fhn_obs,
    "FHN dyn": df_fhn_dyn,
}

#method_list = ["raw", "pca", "features", "features_pca"]
method_list = ["raw", "features", "features_pca"]

signal_colors = {
    "Sine": "C4",
    "FHN obs": "C5",
    "FHN dyn": "C6",
}

alpha_vals = {
    "Train": 1.0,
    "Test": 0.65,
    "Pre": 0.35,
}

# Common x-axis values (sorted union)
all_samples = sorted(
    set(df_sine.periods.unique()) 
    | set(df_fhn_obs.periods.unique()) 
    | set(df_fhn_dyn.periods.unique())
)
x = np.arange(len(all_samples))

# Width of each signal block
group_width = 0.3  # three groups → 0.75 total width
offsets = {
    "Sine": -group_width,
    "FHN obs": 0,
    "FHN dyn": group_width,
}
subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

# ---- Main Loop (one figure per method) ----
for idx, method in enumerate(method_list):

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for signal_name, frame in signal_frames.items():
        color = signal_colors[signal_name]

        dfm = frame[frame["Method"] == method]

        train_means = dfm.set_index("periods")["train_mean"].reindex(all_samples, fill_value=0)
        test_means  = dfm.set_index("periods")["test_mean"].reindex(all_samples, fill_value=0)
        pre_means   = dfm.set_index("periods")["pre_mean"].reindex(all_samples, fill_value=0)

        xo = x + offsets[signal_name]

        # Stacked bars (NO labels)
        ax.bar(xo, train_means,
               width=group_width, color=color, alpha=alpha_vals["Train"])
        ax.bar(xo, test_means,
               width=group_width, bottom=train_means,
               color=color, alpha=alpha_vals["Test"])
        ax.bar(xo, pre_means,
               width=group_width, bottom=train_means + test_means,
               color=color, alpha=alpha_vals["Pre"])

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(all_samples, rotation=45)
    ax.set_xlabel(r"Number of periods ($N_{p}$)")
    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel letter (a), (b), (c), ...
    ax.text(
        0.02, 0.95,
        f"({chr(97 + idx)})",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=12,
        va="top",
        ha="left"
    )

    plt.tight_layout()
    plt.savefig(f"notebooks/times/periods_times_{method}.pdf", format="pdf")
    leg = plt.legend(ncol=9, fontsize=10)
    export_legend(leg, filename=f"notebooks/times/legend_times_{method}.pdf")
    plt.show()

'''
# ---- SubPlots (1x3 layout) ----
fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
axs = axs.flatten()

# Set y-limits for each subplot (optional)
row1_ylim = (0, 0.6)   # adjust as needed
axs = axs.flatten()# ---- SubPlots (1x3 layout) ----
fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
axs = axs.flatten()

# Set y-limits for each subplot (optional)
row1_ylim = (0, 0.6)   # adjust as needed
axs = axs.flatten()

# Loop over methods and axes
for idx, (method, ax) in enumerate(zip(method_list[:3], axs)):

    for signal_name, frame in signal_frames.items():

        color = signal_colors[signal_name]
        dfm = frame[frame["Method"] == method]

        train_means = dfm.set_index("periods")["train_mean"].reindex(all_samples, fill_value=0)
        test_means  = dfm.set_index("periods")["test_mean"].reindex(all_samples, fill_value=0)
        pre_means   = dfm.set_index("periods")["pre_mean"].reindex(all_samples, fill_value=0)

        xo = x + offsets[signal_name]

        # Stacked bars
        ax.bar(xo, train_means,
               width=group_width, color=color, alpha=alpha_vals["Train"])
        ax.bar(xo, test_means,
               width=group_width, color=color, alpha=alpha_vals["Test"],
               bottom=train_means)
        ax.bar(xo, pre_means,
               width=group_width, color=color, alpha=alpha_vals["Pre"],
               bottom=train_means + test_means)

    # Subplot letters
    ax.text(0.02, 0.95, f"{subplot_labels[idx]}",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="top",
            ha="left")

    # Grid
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Optional: set individual y-limits if needed
    ax.set_ylim(0, dfm[['train_mean','test_mean','pre_mean']].values.max()*1.2) 
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

# Global X labels
for ax in axs:
    ax.set_xlabel(r"Number of periods ($N_{p}$)")
axs[0].set_ylabel("Time (s)")

# Set xticks
for ax in axs:
    ax.set_xticks(x)
    ax.set_xticklabels(all_samples, rotation=45)

# Global legend
handles = []
labels = []
for sig, col in signal_colors.items():
    for phase, a in alpha_vals.items():
        handles.append(plt.Rectangle((0,0),1,1,color=col,alpha=a))
        labels.append(f"{sig} {phase}")

#fig.legend(handles, labels, ncol=3, loc="upper center", fontsize=10)

plt.tight_layout()
plt.savefig("notebooks/times/periods_times.pdf", format="pdf")
plt.show()

# Loop over methods and axes
for idx, (method, ax) in enumerate(zip(method_list[:3], axs)):

    for signal_name, frame in signal_frames.items():

        color = signal_colors[signal_name]
        dfm = frame[frame["Method"] == method]

        train_means = dfm.set_index("periods")["train_mean"].reindex(all_samples, fill_value=0)
        test_means  = dfm.set_index("periods")["test_mean"].reindex(all_samples, fill_value=0)
        pre_means   = dfm.set_index("periods")["pre_mean"].reindex(all_samples, fill_value=0)

        xo = x + offsets[signal_name]

        # Stacked bars
        ax.bar(xo, train_means,
               width=group_width, color=color, alpha=alpha_vals["Train"])
        ax.bar(xo, test_means,
               width=group_width, color=color, alpha=alpha_vals["Test"],
               bottom=train_means)
        ax.bar(xo, pre_means,
               width=group_width, color=color, alpha=alpha_vals["Pre"],
               bottom=train_means + test_means)

    # Subplot letters
    ax.text(0.02, 0.95, f"{subplot_labels[idx]}",
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="top",
            ha="left")

    # Grid
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Optional: set individual y-limits if needed
    ax.set_ylim(0, dfm[['train_mean','test_mean','pre_mean']].values.max()*1.2) 
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

# Global X labels
for ax in axs:
    ax.set_xlabel(r"Number of periods ($N_{p}$)")
axs[0].set_ylabel("Time (s)")

# Set xticks
for ax in axs:
    ax.set_xticks(x)
    ax.set_xticklabels(all_samples, rotation=45)

# Global legend
handles = []
labels = []
for sig, col in signal_colors.items():
    for phase, a in alpha_vals.items():
        handles.append(plt.Rectangle((0,0),1,1,color=col,alpha=a))
        labels.append(f"{sig} {phase}")

#fig.legend(handles, labels, ncol=3, loc="upper center", fontsize=10)

plt.tight_layout()
plt.savefig("notebooks/times/periods_times.pdf", format="pdf")
plt.show()
'''