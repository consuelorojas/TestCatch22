import pickle
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('report.mplstyle')

# Load results
dyn = 'FFTs/fhn_dyn/noise_fft/dyn_fft_20260211_160221.pkl'
obs = 'FFTs/fhn_obs/noise_fft/obs_fft_20260211_160314.pkl'
sine = 'FFTs/sine/noise_fft/sine_fft_20260204_125921.pkl'

def read_results(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    

def export_legend(legend, filename="legend.eps"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


# Convert to dataframe
df_dyn = read_results(dyn)
df_obs = read_results(obs)
df_sine = read_results(sine)

long_dyn = pd.DataFrame(df_dyn).explode('auc', ignore_index=True)
long_obs = pd.DataFrame(df_obs).explode('auc', ignore_index=True)
long_sine = pd.DataFrame(df_sine).explode('auc', ignore_index=True)

# compute mean and std
# compute mean and std
dyn_stats = (
    long_dyn
    .groupby('noise')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)

obs_stats = (
    long_obs
    .groupby('noise')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)

sine_stats = (
    long_sine
    .groupby('noise')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)


# Figure
plt.figure()
plt.errorbar(
    sine_stats['noise'],
    sine_stats['mean_auc'],
    yerr = sine_stats['std_auc'],
    fmt='o',
    label = 'Sine'
)

plt.errorbar(
    obs_stats['noise'],
    obs_stats['mean_auc'],
    yerr = obs_stats['std_auc'],
    fmt='*',
    label = 'FHN Observed'
)

plt.errorbar(
    dyn_stats['noise'],
    dyn_stats['mean_auc'],
    yerr = dyn_stats['std_auc'],
    fmt='s',
    label = 'FHN Dynamic'
)

plt.xlabel(r"Noise strength $(D)$")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
#plt.xticks(frame_stats['periods'].unique()[::2])
plt.ylim(0.2, 1.1)
plt.xlim(-0.05, 1.03)
plt.text(-0.05, 1.0, "(d)", fontweight='bold', fontsize=14, va='bottom', ha='left')
plt.savefig('FFTs/fhn_dyn/noise_fft/noise_sweep_auc.eps', format='eps')
plt.show()
