import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_fhn = "results/fhn/fhn_times/results_20251124_130931.pkl"
results_fhn_obs = "results/fhn_obs/times/results_20251124_134421.pkl"
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

# --- Plot with error bars ---
markers = {
    'sine': 'o',
    'fhn_dyn': 'S',
    'fhn_obs': '*'
}

method_colors = {
    "raw": "C0", 
    "pca": "C1", 
    "features": "C2", 
    "features_pca": "C3"
}

plt.figure(figsize=(6.4, 4.8))

for method, color in method_colors.items():
    data = df_fhn_dyn[df_fhn_dyn["Method"] == method]
    plt.errorbar(
        data["samples"], data["train_mean"],
        yerr=data["train_std"],
        fmt='s',         # marker style
        color=color,
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        label=method
    )
    data = df_fhn_obs[df_fhn_obs["Method"] == method]
    plt.errorbar(
        data["samples"], data["train_mean"],
        yerr=data["train_std"],
        fmt='*',         # marker style
        color=color,
        capsize=5,          # error bar caps
        alpha=0.7,
        label=method
    )
    data = df_sine[df_sine["Method"] == method]
    plt.errorbar(
        data["samples"], data["train_mean"],
        yerr=data["train_std"],
        fmt='o',         # marker style
        color=color,
        capsize=5,          # error bar caps
        alpha=0.7,
        label=method
    )

plt.xlabel(r"Samples ($N_s$)")
plt.ylabel("Time (ms)")
#plt.legend(ncol=3, loc ="lower right")
plt.grid(True)
plt.xticks(data.samples.unique()[::2])#, rotation=45)
plt.ylim(-0.01, 0.09)
plt.xlim(0, 255)
plt.text(5, 1.0, "(c)", fontweight="bold", fontsize=14, va="bottom", ha="left")
#plt.tight_layout()
plt.savefig(
    "notebooks/times/fhn_times_plot.pdf",
    format = 'pdf',
    dpi=180
)
plt.show()