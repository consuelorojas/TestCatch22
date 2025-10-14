import numpy as np

'''
Create quasiperiodic signals with noise
'''

def quasiperiodic_base(t, args):
    """
    Generate a quasiperiodic signal with optional static Gaussian noise.

    Parameters:
    - t (np.ndarray): Time vector.
    - args (list or tuple): [f1, f2, noise_strength, n_points, n_periods]
        - f1: frequency of the first sine component in Hz
        - f2: frequency of the second sine component in Hz, has to be irrational
        - noise_strength: standard deviation of Gaussian noise (0 for no noise)
        - n_points: number of points per period
        - n_periods: number of periods to generate
    
    Returns:
    - y (np.ndarray): Noisy quasiperiodic signal.
    """

    f1, f2, noise_strength, _, _ = args

    phi1 =  np.random.uniform(0, 2*np.pi)
    phi2 =  np.random.uniform(0, 2*np.pi)
    y = (np.sin(2 * np.pi * f1 * t + phi1) + np.sin(2 * np.pi * f2 * t + phi2)) / 2

    if noise_strength > 0:
        y += noise_strength * np.random.randn(len(t))

    return y

def generate_quasiperiodic_noise_once(args, base_freq = 5, t_shared=None):
    """
    Generate a quasiperiodic signal over a given number of periods with optional noise.

    Parameters:
    - args (list): [f1, f2, noise_strength, n_points, n_periods]
        - f1 (float): Frequency of the first sine component.
        - f2 (float): Frequency of the second sine component, has to be irrational.
        - noise_strength (float): Standard deviation of Gaussian noise (0 for no noise).
        - n_pts (int): Points per period.
        - n_periodos (float): Number of periods.
    
    Returns:
    - t (np.ndarray): Time vector.
    - y (np.ndarray): Noisy quasiperiodic signal.
    """

    f1, f2, _ , n_points, n_periods = args

    if t_shared is None:
        total_points = int(n_points * n_periods)
        duration = n_periods / base_freq
        t_shared = np.linspace(0, duration, total_points, endpoint=False)

    y = quasiperiodic_base(t_shared, args)
    return t_shared, y