import numpy as np

'''
Generate an amplitude-modulated signal.
'''

def modulated_signal(t, args):
    """
    Generate an amplitude-modulated signal with optional static Gaussian noise.

    Parameters:
    - t (np.ndarray): Time vector.
    - args (list or tuple): [f_c, f_m, modulation_index, noise_strength, n_points, n_periods]
        - f_c: carrier frequency in Hz
        - f_m: modulation frequency in Hz
        - modulation_index: depth of modulation (0 to 1)
        - noise_strength: standard deviation of Gaussian noise (0 for no noise)
        - n_points: number of points per period
        - n_periods: number of periods to generate

    Returns:
    - y (np.ndarray): Amplitude-modulated signal.
    """

    f_c, f_m, modulation_index, noise_strength, _, _ = args

    # Generate the carrier and modulating signals
    phi_c = np.random.uniform(0, 2*np.pi)
    carrier = np.sin(2 * np.pi * f_c * t + phi_c)

    phi_m = np.random.uniform(0, 2*np.pi)
    modulator = 1 + modulation_index * np.sin(2 * np.pi * f_m * t + phi_m)

    # Amplitude modulation
    y = carrier * modulator

    if noise_strength > 0:
        y += noise_strength * np.random.randn(len(t))

    return y

def generate_modulated_signal_once(args, base_freq=5, t_shared=None):
    """
    Generate an amplitude-modulated signal over a given number of periods with optional noise.

    Parameters:
    - args (list): [f_c, f_m, modulation_index, noise_strength, n_points, n_periods]
        - f_c: carrier frequency in Hz
        - f_m: modulation frequency in Hz
        - modulation_index: depth of modulation (0 to 1)
        - noise_strength: standard deviation of Gaussian noise (0 for no noise)
        - n_points: points per period
        - n_periods: number of periods
    - base_freq (float): Base frequency to determine total duration.
    - t_shared (np.ndarray or None): Shared time vector. If None, it will
        be generated based on n_points and n_periods.
    Returns:
    - t (np.ndarray): Time vector.
    - y (np.ndarray): Amplitude-modulated signal.
    """

    f_c, f_m, _, _, n_points, n_periods = args
    
    if t_shared is None:
        total_points = int(n_points * n_periods)
        duration = n_periods / base_freq
        t_shared = np.linspace(0, duration, total_points, endpoint=False)

    # Generate the modulated signal
    y = modulated_signal(t_shared, args)

    return t_shared, y