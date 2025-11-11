import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/fhn/fhn_parameter/results_20251103_172752.pkl"
with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    df = entry["b"]
    for method in ["raw", "pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"b": df, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)

# --- Compute mean & std per method/Δf ---
df_grouped = (
    df_results
    .groupby(["Method", "b"])
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
        data["b"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt=marker,         # marker style
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        label=method
    )

plt.xlabel(r"Parameter Difference $(b - b_0)$")
plt.ylabel("AUC")
#plt.legend(ncol=2, loc ="lower left")
plt.grid(True)
plt.ylim(0.2, 1.1)
plt.xlim(-0.01, 0.31)
plt.text(0.0, 1.00, "(a)", fontweight="bold", fontsize=13, va="bottom", ha="left")
plt.tight_layout()
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/fhn/fhn_parameter/param_fhn_errorbars.png",
    format='png', dpi=180
)
plt.show()