import pickle
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('report.mplstyle')

# Load results
dyn = 'FFTs/fhn_dyn/parameter_fft/dyn_fft_20260211_164807.pkl'
obs = 'FFTs/fhn_obs/parameter_fft/obs_fft_20260211_164820.pkl'
sine = 'FFTs/sine/freq_fft/sine_fft_20260203_175834.pkl'

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
    .groupby('df')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)

obs_stats = (
    long_obs
    .groupby('df')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)

sine_stats = (
    long_sine
    .groupby('df')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)


# Figure
plt.figure()
plt.errorbar(
    sine_stats['df'],
    sine_stats['mean_auc'],
    yerr = sine_stats['std_auc'],
    fmt='o',
    label = 'Sine'
)

plt.errorbar(
    obs_stats['df'],
    obs_stats['mean_auc'],
    yerr = obs_stats['std_auc'],
    fmt='*',
    label = 'FHN Observed'
)

plt.errorbar(
    dyn_stats['df'],
    dyn_stats['mean_auc'],
    yerr = dyn_stats['std_auc'],
    fmt='s',
    label = 'FHN Dynamic'
)
plt.xlabel(r"Parameter Difference")
plt.ylabel('AUC')
plt.grid(True)
plt.tight_layout()
plt.xlim(-0.01, 0.23)
plt.ylim(0.0, 1.1)
plt.text(-0.005, 1.0, "(a)", fontweight='bold', fontsize=14, va='bottom', ha='left')
plt.savefig('FFTs/fhn_dyn/parameter_fft/parameter_sweep_auc.eps', format='eps')
plt.show()
