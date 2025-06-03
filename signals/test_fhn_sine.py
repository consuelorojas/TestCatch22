'''
To run this script. Follow the structure below:

project_folder/
├── test_fhn_sin.py
└── signals/
    ├── __init__.py
    ├── fhn.py
    └── sin.py
'''

import numpy as np
import matplotlib.pyplot as plt

from signals.fhn import SDEs_fhn
from signals.sine import sin_noise

def test_fhn_model():
    print("Testing FitzHugh-Nagumo model...")

    # Parameters
    x0 = [0.0, 0.0]
    tmax = 50.0
    dt = 0.01
    args = [0.7, 0.8, 0.08, 0.5, 0.1]  # b0, b1, epsilon, I, noise_strength

    t, v, w = SDEs_fhn(x0, tmax, dt, args)

    # Basic shape checks
    assert len(t) == len(v) == len(w), "Output arrays must be the same length"

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, v, label='v (membrane potential)')
    plt.plot(t, w, label='w (recovery variable)', alpha=0.6)
    plt.title('FitzHugh-Nagumo Model Output')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_sin_noise():
    print("Testing sinusoidal signal generation...")

    # Parameters
    f = 5  # Hz
    phi = 0
    noise_strength = 0.2
    n_pts = 100
    n_periodos = 5
    args = [f, phi, noise_strength]

    t, y = sin_noise(f, n_pts, n_periodos, args)

    # Basic shape checks
    assert len(t) == len(y), "Time and signal arrays must be the same length"

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, label='Noisy Sine Wave')
    plt.title('Noisy Sine Wave Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_fhn_model()
    test_sin_noise()
