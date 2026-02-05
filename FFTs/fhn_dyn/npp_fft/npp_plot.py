import pickle
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('report.mplstyle')

# Load results
results = 'FFTs/fhn_dyn/npp_fft/dyn_fft_20260204_174329.pkl'

with open(results, 'rb') as f:
    all_results =  pickle.load(f)


def export_legend(legend, filename="legend.eps"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


# Convert to dataframe
frame = pd.DataFrame(all_results)
long_frame = frame.explode('auc', ignore_index=True)
long_frame['auc'] = long_frame['auc'].astype(float)

# compute mean and std
frame_stats = (
    long_frame
    .groupby('npp')['auc']
    .agg(mean_auc='mean', std_auc='std')
    .reset_index()
)


# Figure
plt.figure()
plt.errorbar(
    frame_stats['npp'],
    frame_stats['mean_auc'],
    yerr = frame_stats['std_auc'],
    fmt='P'
)

plt.xlabel(r"Number of points per period $(N_{pp})$")
plt.ylabel("AUC")
plt.grid(True)
plt.tight_layout()
plt.xticks(frame_stats['npp'].unique()[::2])
plt.ylim(0.2, 1.1)
plt.savefig('FFTs/fhn_dyn/npp_fft/npp_sweep_auc.pdf')
plt.show()
