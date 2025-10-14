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

# Sweep configuration
rho_c1 = 28  # baseline chaotic case
drho = 4
rho_g = np.arange(30, 40, drho)   # explore larger rho
rho_s = np.linspace(28, 30, 10)   # fine sweep near onset
rho = np.concatenate((rho_s, rho_g))

pseudo_period = 0.75   # estimated pseudo-period for rho=28
npp = 10               # number of points per period
dt = 0.01
step = int(pseudo_period / (npp * dt))  # subsampling step

trans = 50     # transient steps to drop
samples = 100  # number of series per class

noise_strength = 0.1  # no noise

# Output directory
sweep_name = "lorenz/lorenz_parameter"
output_dir = os.path.join("results", sweep_name)
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"results_{timestamp}.pkl")

all_results = []
for i, r in enumerate(tqdm(rho)):
    # Arguments for Lorenz generator: [[x0, y0, z0], sigma, rho, beta, dt, steps, noise_strength]
    args_base = [[1, 1, 1], 10, rho_c1, 8/3, dt, 750, noise_strength]
    args_var  = [[1, 1, 1], 10, r,      8/3, dt, 750, noise_strength]

    X, y = create_labeled_dataset([

        (0, 'lorenz', {'args': args_base, 'coord': 'x'}),
        (1, 'lorenz', {'args': args_var, 'coord': 'x'})
    ],
        n_samples_per_class=samples,
        subsample_step=step,
        transient=trans
    )

    splits = get_kfold_splits(X, y, n_splits=5, stratified=False)
    results = run_experiment(X, y, splits)

    all_results.append({
        'rho': round(r, 3),
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca']
    })

# Save results
with open(output_file, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Sweep complete. Results saved to {output_file}")
