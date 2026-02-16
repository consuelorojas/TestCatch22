import pickle
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('report.mplstyle')

# Load results
dyn = 'FFTs/fhn_dyn/periods_fft/dyn_fft_20260211_163405.pkl'
obs = 'FFTs/fhn_obs/periods_fft/obs_fft_20260211_164550.pkl'
sine = 'FFTs/sine/periods_fft/sine_fft_20260204_115316.pkl'

def read_results(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    


def export_legend(legend, filename="legend_FFTs.eps"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


# Convert to dataframe
df_dyn = read_results(dyn)
df_obs = read_results(obs)
df_sine = read_results(sine)

long_dyn = pd.DataFrame(df_dyn).explode('auc', ignore_index=True)
lond_dyn = long_dyn['auc'] = long_dyn['auc'].astype(float)

long_obs = pd.DataFrame(df_obs).explode('auc', ignore_index=True)
long_obs['auc'] = long_obs['auc'].astype(float)

long_sine = pd.DataFrame(df_sine).explode('auc', ignore_index=True)
long_sine['auc'] = long_sine['auc'].astype(float)


# compute mean and std
dyn_stats = (
    long_dyn
    .groupby('periods')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)

obs_stats = (
    long_obs
    .groupby('periods')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)

sine_stats = (
    long_sine
    .groupby('periods')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)


# Figure
plt.figure()

plt.errorbar(
    sine_stats['periods'],
    sine_stats['mean_auc'],
    yerr = sine_stats['std_auc'],
    fmt='o',
    label = 'Sine'
)

plt.errorbar(
    obs_stats['periods'],#[obs_stats['periods'] < 10],
    obs_stats['mean_auc'],#[obs_stats['periods'] < 10],
    yerr = obs_stats['std_auc'],#[obs_stats['periods'] < 10],
    fmt='*',
    label = 'FHN Observed'
)

plt.errorbar(
    dyn_stats['periods'],
    dyn_stats['mean_auc'],
    yerr = dyn_stats['std_auc'],
    fmt='s',
    label = 'FHN Dynamic'
)


plt.xlabel(r"Number of periods $(N_p)$")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
legend = plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, ncol=3)
export_legend(legend, filename="FFTs/fhn_dyn/periods_fft/legend_periods_fft.pdf")
#plt.xticks(frame_stats['periods'].unique()[::2])
plt.ylim(0.2, 1.1)
plt.text(0.6, 1.0, "(b)", fontweight='bold', fontsize=14, va='bottom', ha='left')
plt.savefig('FFTs/fhn_dyn/periods_fft/period_sweep_auc.eps', format='eps')
plt.show()
