'''
Dispatcher for generating time series signals from various models.
Each signal model must define a 'generate_<model_name>_signal_once(**kwargs)' function.
'''
from fhn import generate_fhn_signal_once
from sine import generate_sine_noise_once

def get_signal_generator(model_name):
    '''
    Return the appropiate generator function for the specified model name.
    
    Parameters:
    - model_name (str): Name of the signal model (e.g., 'fhn', 'sine').

    Returns:
    - function: Signal generator function that takes the appropriate arguments.

    Raises:
    - ValueError: If the model name is not supported.
    '''
    generators = {
        'fhn': generate_fhn_signal_once,
        'sine': generate_sine_noise_once
        # new models can be added here
    }
    
    if model_name not in generators:
        raise ValueError(
            f"Model '{model_name}' is not supported.",
            f"Available models: {', '.join(generators.keys())}"
            )
    
    return generators[model_name]

def generate_signal(model_name: str, n_samples: int, generator_args: dict):
    '''
    Generate a list of signals from the specified model.

    Parameters:
    - model_name (str): Name of the signal model (e.g., 'fhn', 'sine').
    - n_samples (int): Number of signals to generate.
    - signals_args (list): List of arguments for the signal generator function.
    - return_time (bool): If True, return time and signal; otherwise, return only the signal.
    
    Returns:
    - list: List of generated signals.
    '''
    generator_fn = get_signal_generator(model_name)
    
    return [generator_fn(**generator_args) for _ in range(n_samples)]  # returns (t, y)


