import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/fhn/fhn_times/results_20251112_115812.pkl"
with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

# ---- Convert to DataFrame ----
'''
Data structure in all_results:
    dictionary with keys: "samples", "raw", "pca", "features", "features_pca".
    Each key (except "samples") contains a list of lists of times [train, test, pre] for each fold.
    There are 50 folds, so 50 entries per method.
'''

records = []

for entry in all_results:
    df =  entry["samples"]
    for method in ["raw", "pca", "features", "features_pca"]:
        for times in entry[method]:
            records.append({"samples": df, "Method": method, "Train": times[0], "Test": times[1], "Pre": times[2]})

df_results = pd.DataFrame(records)


# --- Compute mean & std per method/Δf ---
df_grouped = (
    df_results
    .groupby(["Method", "samples"])
    .agg(train_mean=("Train", "mean"), train_std=("AUC", "std"),
         test_mean=("Test", "mean"), test_std=("Test", "std"),
         pre_mean=("Pre", "mean"), pre_std=("Pre", "std"))
    .reset_index()
)

# --- Plot with error bars ---
markers = {
    

}


method_colors = {
    "raw": "C0", 
    "pca": "C1", 
    "features": "C2", 
    "features_pca": "C3"
}

plt.figure(figsize=(6.4, 4.8))

for method, color in method_colors.items():
    data = df_grouped[df_grouped["Method"] == method]
    plt.errorbar(
        data["samples"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt='s',         # marker style
        color=color,
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        label=method
    )

plt.xlabel(r"Samples ($N_s$)")
plt.ylabel("Time (ms)")
#plt.legend(ncol=2, loc ="lower right")
plt.grid(True)
plt.xticks(data.samples.unique()[::2])#, rotation=45)
plt.ylim(0.2, 1.1)
plt.xlim(0, 255)
plt.text(5, 1.0, "(c)", fontweight="bold", fontsize=14, va="bottom", ha="left")
plt.tight_layout()
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/fhn/fhn_samples/samples_fhn_errorbars.eps",
    format = 'eps',
    dpi=180
)
plt.show()