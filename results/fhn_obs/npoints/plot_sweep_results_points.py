import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/fhn_obs/npoints/results_20251103_183951.pkl"
with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    df = entry["npp"]
    for method in ["raw", "pca", "features", "features_pca"]:
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
    "pca": "s", 
    "features": "D", 
    "features_pca": "^"
}

plt.figure(figsize=(6.4, 4.8))

for method, marker in markers.items():
    data = df_grouped[df_grouped["Method"] == method]
    plt.errorbar(
        data["npp"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt=marker,         # marker style
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        label=method
    )

plt.xlabel(r"Number of points per period $(N_{pp})$")
plt.ylabel("AUC")
#plt.legend(ncol=2, loc ="lower left")
plt.grid(True)
plt.tight_layout()
plt.ylim(0.2, 1.1)
plt.xticks(data.npp.unique()[::2])
plt.text(3.0, 1.0, "(c)", fontweight="bold", fontsize=14, va="bottom", ha="left")
#plt.xlim(-0.05, 0.65)
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/fhn_obs/npoints/npp_fhn_errorbars.eps",
    format="eps", dpi=180
)
plt.show()