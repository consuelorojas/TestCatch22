'''
To run this script. Follow the structure below:

project_folder/
├── test_fhn_sin.py
└── signals/
    ├── __init__.py
    ├── fhn.py
    └── sin.py
'''

### this no longer works, at least not with the dispatcher ###

import numpy as np
import matplotlib.pyplot as plt


from fhn import SDEs_fhn
from sine import generate_sine_noise_once as sin_noise
from lorenz import lorenz_base


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

    t, y = sin_noise(args)

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

def test_lorenz_model():
    print("Testing Lorenz attractor model...")

    from lorenz import lorenz_base

    # Parameters
    state0 = [1.0, 1.0, 1.0]
    sigma = 10.0
    rho = 28.0
    beta = 8/3
    dt = 0.01
    steps = 10000
    noise_strength = 20.0
    args = [state0, sigma, rho, beta, dt, steps, noise_strength]

    times, traj = lorenz_base(args)

    # Basic shape checks
    assert traj.shape == (steps, 3), "Trajectory shape must be (steps, 3)"

    # Plot - 3D trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2])
    ax.set_title('Lorenz Attractor')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

    # 1D trajectory plot
    plt.figure(figsize=(10, 4))
    plt.plot(times, traj[:,0], label='X coordinate')
    plt.plot(times, traj[:,1], label='Y coordinate')
    plt.plot(times, traj[:,2], label='Z coordinate')
    plt.title('Lorenz Attractor - X Coordinate Over Time')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #test_fhn_model()
    #test_sin_noise()
    test_lorenz_model()
