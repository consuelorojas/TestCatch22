import pickle
import pandas as pd

with open("all_statistics.pkl", "rb") as f:
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
