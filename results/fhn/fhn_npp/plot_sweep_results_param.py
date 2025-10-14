import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Replace this with your actual path:
result_file = "results/fhn/fhn_npp/results_20251010_150819.pkl"
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

# ---- Boxplot ----
plt.figure(figsize=(20, 14))
sns.boxplot(data=df_results, x="npp", y="AUC", hue="Method", palette="Set2")
#plt.title("AUC across frequency differences by method")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, axis="y")
plt.legend(title="Method")
plt.ylim(-0.1, 1.1)
plt.show()

# ---- Scatter plot with markers ----
markers = {
    "raw": "o", 
    "pca": "s", 
    "features": "D", 
    "features_pca": "^"
}

plt.figure(figsize=(10, 6))

for method, marker in markers.items():
    data = df_results[df_results["Method"] == method]
    plt.scatter(
        data["npp"], data["AUC"],
        label=method,
        marker=marker,
        alpha=0.7
    )

#plt.title("AUC vs Frequency Difference (Δf)")
plt.xlabel(r"Number of points per periods $(N_{pp})$")
plt.ylabel("AUC")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
#plt.xlim(-0.05, 0.65)
plt.savefig("results/fhn/fhn_npp/npp_diff.png", dpi=180)
plt.show()



# --------- scatter plot, lim x axis
plt.figure(figsize=(10, 6))

for method, marker in markers.items():
    data = df_results[df_results["Method"] == method]
    plt.scatter(
        data["npp"], data["AUC"],
        label=method,
        marker=marker,
        alpha=0.7
    )
plt.xlabel(r"Number of points per period $(N_{pp})$")
plt.ylabel("AUC")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
plt.xscale('log')
plt.xticks(data.npp.unique())
#plt.xlim(-0.05, 0.75)
plt.savefig("results/fhn/fhn_npp/npp_diff_lim2.png", dpi=180)
plt.show()

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

plt.figure(figsize=(15, 10))

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
plt.legend(title="Method", loc ="lower left")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
#plt.xlim(-0.05, 0.65)
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/fhn/fhn_npp/npp_fhn_errorbars.png",
    dpi=180
)
plt.show()