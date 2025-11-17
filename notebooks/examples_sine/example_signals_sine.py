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

######### SINE WAVES  #########

## sweep configuration
fbase = 5
f1 = 5.18
nperiods = 3
npoints = 7

noise = 0.1
samples = 100


# Run sweep
X, y = create_labeled_dataset(
    [(0, 'sine', {'args': [fbase, noise, npoints, nperiods]}),
    (1, 'sine', {'args': [f1, noise, npoints, nperiods]})],
    n_samples_per_class=samples
)

plt.figure(figsize=(6.4, 4.8))

t = np.linspace(0, nperiods/fbase, npoints*nperiods)
# Find the indices where y == 0
indices_zero = [i for i in range(len(X)) if y[i] == 0]

# Last one to highlight
highlight_idx = indices_zero[-1]

for i in range(len(X)):
    if y[i] == 0:
        if i == highlight_idx:
            plt.plot(t,X[i], alpha=1, color='tab:blue')  # highlighted last signal
        else:
            plt.plot(t,X[i], marker='', alpha=0.1, color='grey')  # rest are semi-transparent

# Horizontal line across the whole plot at y=0
#plt.axhline(y=0, color='black', marker='',linestyle='-')
plt.ylabel(r"$x(t)$")
plt.xlabel("Time [s]")
plt.title("Class 0: 5.0 Hz")
plt.tight_layout()
plt.savefig('notebooks/examples_sine/example_signals_sine_class0.pdf', format='pdf', dpi=300)

#plt.show()

plt.figure(figsize=(6.4, 4.8))

t2 = np.linspace(0, nperiods/f1, npoints*nperiods)
# Find the indices where y == 1
indices_one = [i for i in range(len(X)) if y[i] == 1]

# Last one to highlight 
highlight_idx = indices_one[-1]

for i in range(len(X)):
    if y[i] == 1:
        if i == highlight_idx:
            plt.plot(t2,X[i], marker='*', alpha=1, color='tab:orange')  # highlighted last signal
        else:
            plt.plot(t2,X[i], marker='', alpha=0.1, color='grey')  # rest are semi-transparent

# Horizontal line across the whole plot at y=1
#plt.axhline(y=0, color='black', marker='',linestyle='-')
plt.ylabel(r"$x(t)$")
plt.xlabel("Time [s]")
plt.title("Class 1: 5.18 Hz")
plt.tight_layout()
plt.savefig('notebooks/examples_sine/example_signals_sine_class1.pdf', format='pdf', dpi=300)
plt.show()

plt.figure(figsize=(6.4, 4.8))

plt.plot(X[indices_one[1]], marker='*', alpha=1, color='tab:red', label='Class 1: 5.18 Hz')
plt.plot(X[indices_zero[1]], alpha=1, color='tab:blue', label='Class 0: 5.0 Hz')
plt.xlabel("Time [s]")
plt.ylabel(r"$x(t)$")
plt.savefig('notebooks/examples_sine/example_signals_sine_single.pdf', format='pdf', dpi=300)
plt.legend(loc = 'lower left')
plt.show()

