import os
import sys
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import run_experiment
from dataset import create_labeled_dataset, get_kfold_splits


## sweep configuration
# Parameters

b0 = 0.1

b1 = 1
db1 = 0.2
b12g = np.arange(1.3, 2.2, db1)
b12s = np.linspace(b1, 1.2, 10)
b12 = np.concatenate((b12s, b12g))
deltab12 = b12 - b1

epsilon = 0.2
I = 0
noise = 0.1

# simulation parameters
dt = 0.1

# step to subsampling
pseudo_period = 30
npp = 10            # change the number of points per period here!
step = int(pseudo_period / npp / dt)

trans = 50 # transient

samples = 100
## Output directory
sweep_name = "fhn_parameter"
output_dir = os.path.join("results", sweep_name)
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"results_{timestamp}.pkl")

# Run sweep
all_results = []
for i, b in enumerate(tqdm(b12)):
    X, y = create_labeled_dataset([
        (0, 'fhn', {'length':750, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
        (1, 'fhn', {'length':750, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b, epsilon, I, noise]})],
        n_samples_per_class=samples, subsample_step = step, transient = trans
        )
    
    splits = get_kfold_splits(X, y, n_splits=5, stratified=False)
    results = run_experiment(X, y, splits)

    all_results.append({
        'b': round(b - b1, 3),
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca']
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")