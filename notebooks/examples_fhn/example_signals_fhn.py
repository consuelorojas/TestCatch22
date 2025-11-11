import os
import sys
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
plt.style.use('report.mplstyle')

sys.path.append(os.path.abspath("./models"))
sys.path.append(os.path.abspath("./data"))

#from classification import run_experiment
from dataset import create_labeled_dataset

######### FITZHUGH-NAGUMO  #########

## sweep configuration
b0 = 0.1

b1 = 1.0
db1 = 0.032
#db1 = 0.175
b12 = b1 + db1
dt = 0.1


# step to subsampling
pseudo_period = 30
npp = 10          # change the number of points per period here!
step = int(pseudo_period / npp / dt)


epsilon = 0.2
I = 0
noise = 0.1

trans = 100 # transient

samples = 80


X, y, t = create_labeled_dataset([
    (0, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b1, epsilon, I, noise]}),
    (1, 'fhn_obs', {'length':850, 'dt': 0.1, 'x0': [0,0], 'args':[b0, b12, epsilon, I, noise]})],
    n_samples_per_class=samples, subsample_step = step, transient = trans, return_time=True
    )


plt.figure(figsize=(6.4, 4.8))

# Find the indices where y == 0
indices_zero = [i for i in range(len(X)) if y[i] == 0]

# Last one to highlight
highlight_idx = indices_zero[-1]

for i in range(len(X)):
    if y[i] == 0:
        if i == highlight_idx:
            plt.plot(t[i],X[i], alpha=1, color='tab:blue')  # highlighted last signal
        else:
            plt.plot(t[i],X[i], marker='', alpha=0.1, color='grey')  # rest are semi-transparent

# Horizontal line across the whole plot at y=0
#plt.axhline(y=0, color='black', marker='',linestyle='-')
plt.ylabel(r"$v(t)$")
plt.xlabel("Time [a.u]")
plt.title("Class 0: b=1.0")
plt.tight_layout()
plt.savefig('notebooks/examples_fhn_obs/example_signals_fhn_obs_class0.pdf', format='pdf', dpi=300)
#plt.show()

plt.figure(figsize=(6.4, 4.8))

# Find the indices where y == 1
indices_one = [i for i in range(len(X)) if y[i] == 1]

# Last one to highlight
highlight_idx = indices_one[-1]

for i in range(len(X)):
    if y[i] == 1:
        if i == highlight_idx:
            plt.plot(t[i],X[i], marker='*', alpha=1, color='tab:orange')  # highlighted last signal
        else:
            plt.plot(t[i],X[i], marker='', alpha=0.1, color='grey')  # rest are semi-transparent

# Horizontal line across the whole plot at y=1
#plt.axhline(y=0, color='black', marker='',linestyle='-')
plt.ylabel(r"$v(t)$")
plt.title(f"Class 1: b={b12}")
plt.xlabel("Time [a.u]")
plt.tight_layout()
plt.savefig('notebooks/examples_fhn_obs/example_signals_fhn_obs_class1.pdf', format='pdf', dpi=300)
#plt.show()

plt.figure(figsize=(6.4, 4.8))
plt.plot(X[indices_one[1]], marker='*', alpha=1, color='tab:red', label=f'Class 1: b={b12}')
plt.plot(X[indices_zero[1]], alpha=1, color='tab:blue', label=f'Class 0: b={b1}')
plt.xlabel("Time [a.u]")
plt.ylabel(r"$v(t)$")
plt.legend(loc = 'lower left')
#plt.savefig('notebooks/examples_fhn/example_signals_fhn_single.png', dpi=300)
plt.show()

