import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Results path:
result_file = "results/sine/sine_frequency/results_20251103_155126.pkl"

with open(result_file, 'rb') as f:
    all_results = pickle.load(f)

def export_legend(legend, filename="legend.eps"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

# ---- Convert to DataFrame ----
records = []
for entry in all_results:
    df = entry["df"]
    for method in ["raw", "pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"Δf": df, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)

'''
# ---- Boxplot ----
plt.figure(figsize=(20, 14))
sns.boxplot(data=df_results, x="Δf", y="AUC", hue="Method", palette="Set2")
plt.title("AUC across frequency differences by method")
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

plt.figure(figsize=(20, 14))

for method, marker in markers.items():
    data = df_results[df_results["Method"] == method]
    plt.scatter(
        data["Δf"], data["AUC"],
        label=method,
        marker=marker,
        alpha=0.7
    )

#plt.title("AUC vs Frequency Difference (Δf)")
plt.xlabel(r"Parameter Difference $(\omega_0 - \omega_1)$")
plt.ylabel("AUC")
plt.legend(title="Method")
plt.grid(True)
plt.tight_layout()
plt.ylim(-0.1, 1.1)
#plt.xlim(-0.05, 0.65)
#plt.savefig("/home/consuelo/Documentos/GitHub/TestCatch22/results/sine/sine_frequency/auc_vs_freq_diff2.png", dpi=180)
plt.show()

'''

# --- Compute mean & std per method/Δf ---
df_grouped = (
    df_results
    .groupby(["Method", "Δf"])
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

method_labels = {
    "raw": "Raw",
    "pca": "PCA",
    "features": "Catch22",
    "features_pca": "Catch22 + PCA"
}

plt.figure(figsize=(6.4, 4.8))

for method, marker in markers.items():
    data = df_grouped[df_grouped["Method"] == method]
    plt.errorbar(
        data["Δf"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt=marker,         # marker style
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        label=method_labels[method]
    )

plt.xlabel(r"Parameter Difference $(\nu - \nu_1)$")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
plt.ylim(0.2, 1.1)
plt.text(0.0, 1.0, "(a)", fontweight="bold", fontsize=14, va="bottom", ha="left")
#plt.xlim(-0.05, 0.65)
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/sine/sine_frequency/errorbars_0-042.eps",
    format="eps", dpi=180
)
#legend = plt.legend(fontsize=14,ncol=1)
#export_legend(legend)
plt.show()



