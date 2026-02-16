import pickle
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('report.mplstyle')

# Load results
dyn = 'FFTs/fhn_dyn/npp_fft/dyn_fft_20260204_174329.pkl'
obs = 'FFTs/fhn_obs/npp_fft/obs_fft_20260204_175604.pkl'
sine = 'FFTs/sine/points_fft/sine_fft_20260204_103857.pkl'

def read_results(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
# Convert to dataframe
df_dyn = read_results(dyn)
df_obs = read_results(obs)
df_sine = read_results(sine)

long_dyn = pd.DataFrame(df_dyn).explode('auc', ignore_index=True)
long_obs = pd.DataFrame(df_obs).explode('auc', ignore_index=True)
long_sine = pd.DataFrame(df_sine).explode('auc', ignore_index=True)

# compute mean and std
dyn_stats = (
    long_dyn
    .groupby('npp')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)

obs_stats = (
    long_obs
    .groupby('npp')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)

sine_stats = (
    long_sine
    .groupby('npp')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)
# Figure
plt.figure()
plt.figure()
plt.errorbar(
    sine_stats['npp'],
    sine_stats['mean_auc'],
    yerr = sine_stats['std_auc'],
    fmt='o',
    label = 'Sine'
)

plt.errorbar(
    obs_stats['npp'],
    obs_stats['mean_auc'],
    yerr = obs_stats['std_auc'],
    fmt='*',
    label = 'FHN Observed'
)

plt.errorbar(
    dyn_stats['npp'],
    dyn_stats['mean_auc'],
    yerr = dyn_stats['std_auc'],
    fmt='s',
    label = 'FHN Dynamic'
)

plt.xlabel(r"Number of points per period $(N_{pp})$")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
plt.xticks(sine_stats['npp'].unique()[::2])
plt.ylim(0.2, 1.1)
plt.text(0.6, 1.0, "(c)", fontweight='bold', fontsize=14, va='bottom', ha='left')
plt.savefig('FFTs/fhn_dyn/npp_fft/npp_sweep_auc.eps', format='eps')
plt.show()
