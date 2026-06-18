import pickle
import pandas as pd

with open("notebooks/all_statistics.pkl", "rb") as f:
    all_statistics = pickle.load(f)

# %%
records = []

for signal, experiments in all_statistics.items():

    for exp_name, stats in experiments.items():

        if "error" in stats:
            continue

        records.append({
            "signal": signal,
            "experiment": exp_name,
            "friedman_stat": stats["friedman_stat"],
            "friedman_p": stats["friedman_p"],
            "n_blocks": stats["n_blocks"]
        })

df_summary = pd.DataFrame(records)

# %%
posthoc_records = []

for signal, experiments in all_statistics.items():

    for exp_name, stats in experiments.items():

        if "error" in stats:
            continue

        df_posthoc = stats["posthoc"].copy()

        df_posthoc["signal"] = signal
        df_posthoc["experiment"] = exp_name

        posthoc_records.append(df_posthoc)

df_posthoc_all = pd.concat(posthoc_records, ignore_index=True)


# %%
# %%
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("/home/consuelo/Documentos/GitHub/TestCatch22/report.mplstyle")


df = df_posthoc_all.copy()

# -----------------------------
# 1. Collapse experiments
# -----------------------------
df["experiment"] = df["experiment"].replace({
    "b": "param",
    "df": "param"
})

# -----------------------------
# 2. Clean method names
# -----------------------------
label_map = {
    "features": "Feature",
    "features_pca": "Feature + PCA",
    "fft": "FFT",
    "fft_pca": "FFT + PCA",
    "pca": "PCA",
    "raw": "Raw"
}

df["Method_1"] = df["Method_1"].map(label_map).fillna(df["Method_1"])
df["Method_2"] = df["Method_2"].map(label_map).fillna(df["Method_2"])

# -----------------------------
# 3. Create unordered pair labels
# -----------------------------
df["pair"] = df.apply(
    lambda x: " vs ".join(sorted([x["Method_1"], x["Method_2"]])),
    axis=1
)

# -----------------------------
# 4. Aggregate NON-significant counts
# -----------------------------
pivot = df.groupby(["experiment", "pair"])["significant"] \
          .apply(lambda x: (~x).sum()) \
          .reset_index()

heatmap = pivot.pivot(index="experiment", columns="pair", values="significant").fillna(0)

# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(12, 5))

plt.imshow(heatmap, aspect="auto", cmap='viridis')

plt.xticks(range(len(heatmap.columns)), heatmap.columns, rotation=45, ha="right")
plt.yticks(range(len(heatmap.index)), heatmap.index)

plt.colorbar(label="Non-significant comparisons")

plt.title("Method equivalence structure (Holm-corrected Wilcoxon)")
plt.tight_layout()
plt.grid(False)

plt.show()