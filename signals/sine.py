
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
    - args (list or tuple): [f, noise_strength, n_points, n_periods]
        - f: frequency in Hz
        - noise_strength: standard deviation of Gaussian noise (0 for no noise)
        - n_points: number of points per period
        - n_periods: number of periods to generate

    Returns:
    - y (np.ndarray): Noisy sine wave.
    """
    f, noise_strength, _, _ = args
    phi =  np.random.uniform(0, 2*np.pi)
    y = np.sin(2 * np.pi * f * t + phi)
    if noise_strength > 0:
        y += noise_strength * np.random.randn(len(t))
    return y


def generate_sine_noise_once(args):
    """
    Generate a sine wave over a given number of periods with optional noise.

    Parameters:
    - args (list): [f, noise_strength, n_points, n_periods]
        - f (float): Frequency of the sine wave.
        - noise_strength (float): Standard deviation of Gaussian noise (0 for no noise).
        - n_pts (int): Points per period.
        - n_periodos (float): Number of periods.
    

    Returns:
    - t (np.ndarray): Time vector.
    - y (np.ndarray): Noisy sine wave.
    """
    f, _ , n_points, n_periods= args
    total_points = int(n_points * n_periods)
    duration = n_periods / f
    t = np.linspace(0, duration, total_points, endpoint=False)
    y = sinusoidal_base(t, args)

    return t, y
