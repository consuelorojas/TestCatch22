import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('report.mplstyle')

# ---- Load results from file ----
# Results path:
result_file = "results/sine/sine_frequency/results_20260224_145230.pkl"

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
    for method in ["raw", "pca", "fft", "fft_pca", "features", "features_pca"]:
        for auc in entry[method]:
            records.append({"Δf": df, "Method": method, "AUC": auc})

df_results = pd.DataFrame(records)

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
    "fft": "s",
    "fft_pca": "P",
    #"pca": "s", 
    "features": "D", 
    "features_pca": "^"
}

method_colors = {
    "raw": "C0", 
    "fft": "C1",
    "fft_pca": "C4",
    #"pca": "C1", 
    "features": "C2", 
    "features_pca": "C3"
}

method_labels = {
    "raw": "Raw",
    "fft": "FFT",
    "fft_pca": "FFT + PCA",
    "features": "Catch22",
    "features_pca": "Catch22 + PCA"
}

plt.figure(figsize=(6.4, 4.8))

for method, color in method_colors.items():
    data = df_grouped[df_grouped["Method"] == method]
    plt.errorbar(
        data["Δf"], data["AUC_mean"],
        yerr=data["AUC_std"],
        fmt='o',       # marker style
        capsize=5,          # error bar caps
        #elinewidth=1,       # error bar line thickness
        alpha=0.7,
        label=method_labels[method],
        color=color
    )

plt.xlabel(r"Parameter Difference $(\nu - \nu_1)$")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
plt.ylim(0.0, 1.1)
plt.text(0.0, 1.0, "(a)", fontweight="bold", fontsize=14, va="bottom", ha="left")
#plt.xlim(-0.05, 0.65)
plt.savefig(
    "/home/consuelo/Documentos/GitHub/TestCatch22/results/sine/sine_frequency/errorbars_0-042.eps",
    format="eps", dpi=180
)
#legend = plt.legend(fontsize=14,ncol=5)
#export_legend(legend)
plt.show()



