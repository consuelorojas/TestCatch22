import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_fhn = "results/fhn/fhn_times/results_20251124_130931.pkl"
results_fhn_obs = "results/fhn_obs/times/results_20251125_151259.pkl"
results_sine = "results/sine/sine_times/results_20251124_145817.pkl"



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
        df =  entry["samples"]
        for method in ["raw", "pca", "features", "features_pca"]:
            for times in entry[method]:
                records.append({"samples": df, "Method": method, "Train": times[0], "Test": times[1], "Pre": times[2]})

    df_results = pd.DataFrame(records)
    return df_results

sine_results = files_to_dataframe(results_sine)
fhn_dyn_results = files_to_dataframe(result_fhn)
fhn_obs_results = files_to_dataframe(results_fhn_obs)


# --- Compute mean & std per method/Δf ---
df_sine = (
    sine_results
    .groupby(["Method", "samples"])
    .agg(train_mean=("Train", "mean"), train_std=("Train", "std"),
         test_mean=("Test", "mean"), test_std=("Test", "std"),
         pre_mean=("Pre", "mean"), pre_std=("Pre", "std"))
    .reset_index()
)
df_fhn_dyn = (
    fhn_dyn_results
    .groupby(["Method", "samples"])
    .agg(train_mean=("Train", "mean"), train_std=("Train", "std"),
         test_mean=("Test", "mean"), test_std=("Test", "std"),
         pre_mean=("Pre", "mean"), pre_std=("Pre", "std")) 
    .reset_index()
)
df_fhn_obs = (
    fhn_obs_results
    .groupby(["Method", "samples"])
    .agg(train_mean=("Train", "mean"), train_std=("Train", "std"),
         test_mean=("Test", "mean"), test_std=("Test", "std"),
         pre_mean=("Pre", "mean"), pre_std=("Pre", "std")) 
    .reset_index()
)

# ---- Plot Config ----
signal_frames = {
    "Sine": df_sine,
    "FHN dyn": df_fhn_dyn,
    "FHN obs": df_fhn_obs,
}

method_list = ["raw", "pca", "features", "features_pca"]

signal_colors = {
    "Sine": "C0",
    "FHN dyn": "C1",
    "FHN obs": "C2",
}

alpha_vals = {
    "Train": 1.0,
    "Test": 0.65,
    "Pre": 0.35,
}

# Common x-axis values (sorted union)
all_samples = sorted(
    set(df_sine.samples.unique()) 
    | set(df_fhn_dyn.samples.unique()) 
    | set(df_fhn_obs.samples.unique())
)[::4]
x = np.arange(len(all_samples))

# Width of each signal block
group_width = 0.3  # three groups → 0.75 total width
offsets = {
    "Sine": -group_width,
    "FHN dyn": 0,
    "FHN obs": group_width,
}
subplot_labels = ["(e)", "(f)", "(g)", "(h)"]

# ---- Main Loop (one figure per method) ----
for method in method_list:

    fig, ax = plt.subplots(figsize=(11, 5))

    for signal_name, frame in signal_frames.items():
        color = signal_colors[signal_name]

        dfm = frame[frame["Method"] == method]

        # align times to master sample list
        train_means = dfm.set_index("samples")["train_mean"].reindex(all_samples, fill_value=0)
        test_means  = dfm.set_index("samples")["test_mean"].reindex(all_samples, fill_value=0)
        pre_means   = dfm.set_index("samples")["pre_mean"].reindex(all_samples, fill_value=0)

        # Offsetting signal types
        xo = x + offsets[signal_name]

        # Stacked bars
        ax.bar(xo, train_means, 
               width=group_width, 
               color=color, 
               alpha=alpha_vals["Train"], 
               label=f"{signal_name} Train" if method == method_list[0] else None)

        ax.bar(xo, test_means, 
               width=group_width, 
               bottom=train_means, 
               color=color, 
               alpha=alpha_vals["Test"], 
               label=f"{signal_name} Test" if method == method_list[0] else None)

        ax.bar(xo, pre_means, 
               width=group_width, 
               bottom=train_means + test_means,
               color=color, 
               alpha=alpha_vals["Pre"], 
               label=f"{signal_name} Pre" if method == method_list[0] else None)

    # Labels and formatting
    ax.set_title(f"Timing Breakdown – {method}")
    ax.set_xticks(x)
    ax.set_xticklabels(all_samples, rotation=45)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    #ax.xlim(0, 530)

    # only show legend in first plot
    if method == method_list[0]:
        ax.legend(ncol=3, fontsize=9)

    plt.tight_layout()
    plt.show()

# ---- SubPlots

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axs = axs.flatten()

for idx, (method, ax) in enumerate(zip(method_list, axs)):

    for signal_name, frame in signal_frames.items():

        color = signal_colors[signal_name]
        dfm = frame[frame["Method"] == method]

        train_means = dfm.set_index("samples")["train_mean"].reindex(all_samples, fill_value=0)
        test_means  = dfm.set_index("samples")["test_mean"].reindex(all_samples, fill_value=0)
        pre_means   = dfm.set_index("samples")["pre_mean"].reindex(all_samples, fill_value=0)

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

    # Titles + subplot letters
    ax.set_title(f"{subplot_labels[idx]}  {method}", loc="left", fontsize=12)

    # Grid
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# Global X/Y labels
axs[2].set_xlabel("Samples (Nₛ)")
axs[3].set_xlabel("Samples (Nₛ)")
axs[0].set_ylabel("Time (s)")
axs[2].set_ylabel("Time (s)")

# Set xticks only once
for ax in axs:
    ax.set_xticks(x)
    ax.set_xticklabels(all_samples, rotation=45)

# Global legend (from first axis handles)
handles = []
labels = []
for sig, col in signal_colors.items():
    for phase, a in alpha_vals.items():
        handles.append(
            plt.Rectangle((0,0),1,1, color=col, alpha=a)
        )
        labels.append(f"{sig} {phase}")

#fig.legend(handles, labels, ncol=9, loc="upper center", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()