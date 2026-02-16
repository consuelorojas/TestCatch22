import os
import sys
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from dataset import create_labeled_dataset, get_kfold_splits

## sweep configuration
b0 = 0.1

b1 = 1
db1 = 0.18
b12 = b1 + db1

epsilon = 0.2
I = 0
noise = 0
dt = 0.1

# step to subsampling
pseudo_period = 30
npp = 13          # change the number of points per period here!
step = int(pseudo_period / npp / dt)


epsilon = 0.2
I = 0
noise = np.arange(0, 0.5, 0.05)
trans = 50 # transient

samples = 100
## Output directory
sweep_name = "fhn_dyn/noise_fft"
output_dir = os.path.join("FFTs", sweep_name)
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"dyn_fft_{timestamp}.pkl")

# Run sweep
all_results = []
for i, n in enumerate(tqdm(noise)):
    X, y = create_labeled_dataset([ #type: ignore
        (0, 'fhn', {'length':750, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, n]}),
        (1, 'fhn', {'length':750, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b12, epsilon, I, n]})],
        n_samples_per_class=samples, subsample_step = step, transient = trans
        )
    
    # compute one-dimensional discrete fourier transform of the matrix X with real inputs
    X_fft = np.abs(np.fft.rfft(X, axis=1))

    splits = get_kfold_splits(X_fft, y, n_splits=50, stratified=True)
    auc_scores = []
    for train_idx, test_idx in splits:

        x_train, x_test = X_fft[train_idx], X_fft[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = SVC(probability=True, random_state=42)
        clf.fit(x_train, y_train)

        y_pred = clf.predict_proba(x_test)[:, 1]
        auc_scores.append(roc_auc_score(y_test, y_pred))
    all_results.append({
        'noise': noise[i],
        'auc': auc_scores
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")
