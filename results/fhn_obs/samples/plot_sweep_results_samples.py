import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/fhn_obs/samples/results_20260304_105933.pkl"
with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    df = entry["samples"]
    for method in ["raw", "pca", "fft", "fft_pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"samples": df, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)

# --- Compute mean & std per method/Δf ---
df_grouped = (
    df_results
    .groupby(["Method", "samples"])
    .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"))
    .reset_index()
)

# --- Plot with error bars ---
markers = {
    "raw": "o", 
    "fft": "s",
    "fft_pca": "P",
    "pca": "s", 
    "features": "D", 
    "features_pca": "^"
}

method_colors = {
    "raw": "C0",
    "pca": "C1", 
    "fft": "C2",
    "fft_pca": "C3",
    "features": "C4", 
    "features_pca": "C5"
}


plt.figure(figsize=(6.4, 4.8))

for method, color in method_colors.items():
    data = df_grouped[df_grouped["Method"] == method]
    plt.errorbar(
        data["samples"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt='*',
        markersize=10,         # marker style
        color=color,
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        #label=method_labels[method]
    )
#print(data.samples.unique())
plt.xlabel(r"Samples ($N_s$)")
plt.ylabel("AUC")
#plt.legend(ncol=2, loc ="lower left")
plt.grid(True)
plt.xticks(data.samples.unique()[::2])
plt.ylim(0., 1.1)
plt.xlim(0, 255)
plt.text(-0.13, 1.01, "(b)", fontweight="bold", fontsize=13, va="bottom", ha="left", transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/fhn_obs/samples/samples_fhn_obs_errorbars.eps",
    format="eps", dpi=180
)
plt.show()