import os
import sys
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

from classification import run_experiment
from dataset import create_labeled_dataset, get_kfold_splits


## sweep configuration
b0 = 0.1

b1 = 1
db1 = 0.18
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

samples = np.arange(10, 520, 10)
## Output directory
sweep_name = "fhn/fhn_samples"
output_dir = os.path.join("results", sweep_name)
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"results_{timestamp}.pkl")

# Run sweep
def run_single_experiment(sample):
    X, y = create_labeled_dataset([
        (0, 'fhn', {'length':750, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
        (1, 'fhn', {'length':750, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b12, epsilon, I, noise]})],
        n_samples_per_class=sample, subsample_step = step, transient = trans
        )

    splits = get_kfold_splits(X, y, n_splits=50, stratified=True)
    #print(f" train set size: {len(splits[0][0])}, test set size: {len(splits[0][1])}")
    results = run_experiment(X, y, splits, ffts=True)
    

    return{
        'samples': len(X) // 2,
        'raw': results['raw'],
        'pca': results['pca'],
        'features': results['features'],
        'features_pca': results['features_pca'],
        'fft': results['fft'],
        'fft_pca': results['fft_pca']
    }

# Save results
def main():
    all_results = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_single_experiment, sample): sample for sample in samples
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Sweeping number of samples"
        ):
            all_results.append(future.result())

    # save results
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Sweep complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()