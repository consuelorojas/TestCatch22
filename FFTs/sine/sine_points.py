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
fbase = 5
f1 = 5.18
npoints = np.arange(1, 21)  # From 1 to 20 in steps of 2
nperiods = 3

noise = 0.1
samples = 100

sweep_name = "sine/points_fft"
output_dir = os.path.join("FFTs", sweep_name)
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"sine_fft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

# Run sweep
all_results = []
for i, npp in enumerate(tqdm(npoints, desc="Sweeping number of points")):
    X, y = create_labeled_dataset( # type: ignore
        [(0, 'sine', {'args': [fbase, 0.1, npp, nperiods]}),
         (1, 'sine', {'args': [f1, 0.1, npp, nperiods]})],
        n_samples_per_class=samples
    )

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
        #print(f"npp: {npp}, Fold AUC: {auc_scores[-1]:.3f}")
    all_results.append({
        'npp': npp,
        'auc': auc_scores
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")

