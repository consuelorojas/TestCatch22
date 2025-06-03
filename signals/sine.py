
import numpy as np
import random

'''
Create sinusoidal signals with noise
'''

def sinusoidal_base(t, args):
    """
    Generate a sine wave with optional static Gaussian noise.

    Parameters:
    - t (np.ndarray): Time vector.
    - args (list or tuple): [f, phi, noise_strength]
        - f: frequency in Hz
        - phi: phase in radians
        - noise_strength: standard deviation of Gaussian noise (0 for no noise)

    Returns:
    - y (np.ndarray): Noisy sine wave.
    """
    f, _, noise_strength = args
    phi =  np.random.uniform(0, 2*np.pi)
    y = np.sin(2 * np.pi * f * t + phi)
    if noise_strength > 0:
        y += noise_strength * np.random.randn(len(t))
    return y


def generate_sine_noise_once(n_points, n_periods, args):
    """
    Generate a sine wave over a given number of periods with optional noise.

    Parameters:
    - f (float): Frequency of the sine wave.
    - n_pts (int): Points per period.
    - n_periodos (float): Number of periods.
    - args (list): [f, phi, noise_strength]

    Returns:
    - t (np.ndarray): Time vector.
    - y (np.ndarray): Noisy sine wave.
    """
    f, _, _ = args
    total_points = int(n_points * n_periods)
    duration = n_periods / f
    t = np.linspace(0, duration, total_points, endpoint=False)
    y = sinusoidal_base(t, args)

    return t, y
