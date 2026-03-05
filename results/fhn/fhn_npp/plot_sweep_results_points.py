import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/fhn/fhn_npp/results_20260224_163538.pkl"
with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    df = entry["npp"]
    for method in ["raw", "fft", "fft_pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"npp": df, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)


# --- Compute mean & std per method/Δf ---
df_grouped = (
    df_results
    .groupby(["Method", "npp"])
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
#    "pca": "C1", 
    "fft": "C2",
    "fft_pca": "C3",
    "features": "C4", 
    "features_pca": "C5"
}

method_labels = {
    "raw": "Raw",
    "pca": "Raw + PCA",
    "fft": "FFT",
    "fft_pca": "FFT + PCA",
    "features": "Catch22",
    "features_pca": "Catch22 + PCA"
}

plt.figure(figsize=(6.4, 4.8))

for method, color in method_colors.items():
    data = df_grouped[df_grouped["Method"] == method]
    plt.errorbar(
        data["npp"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt='s',         # marker style
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        label=method,
        color=color
    )

plt.xlabel(r"Number of points per period $(N_{pp})$")
plt.ylabel("AUC")
#plt.legend(ncol=2, loc ="lower left")
plt.grid(True)
plt.ylim(0.4, 1.05)
plt.text(-0.13, 1.01, "(c)", fontweight="bold", fontsize=14, va="bottom", ha="left", transform=plt.gca().transAxes)
plt.xticks(data.npp.unique()[::2])
plt.tight_layout()
#plt.xlim(-0.05, 0.65)
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/fhn/fhn_npp/npp_fhn_errorbars.eps",
    format='eps', dpi=180
)
plt.show()