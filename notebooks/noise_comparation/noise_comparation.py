import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

plt.style.use('report.mplstyle')

# load results
sine = "results/sine/sine_noise/results_20251112_112335.pkl"
fhn_obs = "results/fhn_obs/noise/results_20251103_183428.pkl"
fhn_dyn = "results/fhn/fhn_noise/results_20251112_115626.pkl"

def extract_results(results):
    with open(results, "rb") as f:
        results = pickle.load(f)
    records = []
    for entry in results:
        noise_level = entry["noise"]
        for method in ["raw", "pca", "features", "features_pca"]:
            for auc in entry[method]:
                records.append({"noise": noise_level, "Method": method, "AUC": auc})
    return pd.DataFrame(records)

def export_legend(legend, filename="legend.eps"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

df_sine = extract_results(sine)
df_fhn_obs = extract_results(fhn_obs)
df_fhn_dyn = extract_results(fhn_dyn)

noise = sorted(df_sine["noise"].unique())
methods = ["raw", "pca", "features", "features_pca"]
panel = ["(a)", "(b)", "(c)", "(d)"]

markers = {
    "sine": "o",
    "fhn_obs": "*",
    "fhn_dyn": "s"
}
labels = {
    "sine": "Sine",
    "fhn_obs": "FHN Observed",
    "fhn_dyn": "FHN Dynamic"
}

method_colors = {
    "raw": "C0", 
    "pca": "C1", 
    "features": "C2", 
    "features_pca": "C3"
}

sine_grouped = (
    df_sine
    .groupby(["Method", "noise"])
    .agg(AUC_mean=("AUC","mean"), AUC_std=("AUC", "std"))
    .reset_index()
)
fhn_obs_grouped = (
    df_fhn_obs
    .groupby(["Method", "noise"])
    .agg(AUC_mean=("AUC","mean"), AUC_std=("AUC", "std"))
    .reset_index()
)
fhn_dyn_grouped = (
    df_fhn_dyn
    .groupby(["Method", "noise"])
    .agg(AUC_mean=("AUC","mean"), AUC_std=("AUC", "std"))
    .reset_index()
)
frames = [sine_grouped, fhn_obs_grouped, fhn_dyn_grouped]


for i, method in enumerate(methods):
    plt.figure(figsize=(6.4,4.8))
    for df, label in zip(frames, markers.keys()):
        data = df[df["Method"] == method]
        plt.errorbar(
            data["noise"], data["AUC_mean"], yerr=data["AUC_std"],
            label=labels[label],
            marker=markers[label],
            linestyle='',
            capsize=5,
            color = method_colors[method],
        )
    plt.xlabel(r"Noise Strength $(D)$")
    plt.ylabel("AUC")
    plt.ylim(0.2, 1.1)
    #plt.title(f"AUC vs Noise Level for {method}")
    plt.xlim(-0.05, 0.65)
    plt.text(-0.05, 1.03, f"{panel[i]}",
             fontweight="bold",
             fontsize=14,
             va="bottom",
             ha="left")
    plt.tight_layout()
    plt.savefig(f"notebooks/noise_comparation/{method}.eps",
                format= "eps")
    #legend = plt.legend(fontsize=14, ncol=3)
    #export_legend(legend, filename="noise_legend.eps")
    
    plt.show()
    