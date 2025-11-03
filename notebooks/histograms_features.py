import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from sklearn.svm import SVC
from pycatch22 import catch22_all
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



plt.style.use('report.mplstyle')

# own modules
sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))
sys.path.append(os.path.abspath("./features"))
sys.path.append(os.path.abspath("./preprocessing"))
from dataset import create_labeled_dataset, get_kfold_splits
from features import extract_features
from preprocessing import apply_pca

cmap = mpl.colormaps.get_cmap("coolwarm").with_extremes(under="w")
cmap.set_bad("gray")



## sweep configuration
b0 = 0.1

b1 = 1
db1 = 0.178
b12 = b1 + db1

epsilon = 0.2
I = 0
noise = 0.1
dt = 0.1

# step to subsampling
pseudo_period = 30
npp = 10          # change the number of points per period here!
step = int(pseudo_period / npp / dt)


epsilon = 0.2
I = 0
trans = 50 # transient
samples = 100

X, y, t = create_labeled_dataset([
    (0, 'fhn', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
    (1, 'fhn', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b12, epsilon, I, noise]})],
    n_samples_per_class=samples, subsample_step = step, transient = trans, return_time=True
    )

features = extract_features(X)
features_df = pd.DataFrame(features)
features_df['label'] = y

plot_df = features_df.melt(id_vars='label', value_vars=['acf_timescale', 'low_freq_power'],
                           var_name = 'features', value_name='value' )

plt.figure(figsize=(8,6))
sns.histplot(data=plot_df, x='value', hue='label', bins=30, kde=True)
plt.xlabel('Feature value')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

