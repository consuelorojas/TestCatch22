import numpy as np
import matplotlib.pyplot as plt

'''
Create signals using the FitzHugh-Nagumo model
'''

def fitzhugh_nagumo(x, args):

    '''
    FitzHugh-Nagumo model of neural activity

    Parameters:
    - x: state vector [v, w]
    - args: parameters [b0, b1, epsilon, I]
    '''

    # unpack parameters
    b0, b1, epsilon, I, _ = args
    v, w = x

    # define the system of equations
    dvdt = v - (1/3)*v**3 - w + I
    dwdt = epsilon*(b0 + b1*v - w)

    return np.array([dvdt, dwdt])



def SDEs_fhn(x0, tmax, dt, args):
    
    '''
    SDEs for FitzHugh-Nagumo model

    Parameters:
    - x0: initial state [v, w]
    - tmax: maximum time to simulate
    - dt: time step
    - args: parameters [b0, b1, epsilon, I, noise_strength]
    '''
    _, _, _, _, noise_strength = args
    v0 = np.random.normal(0.0, 1.5, 1)
    w0 = np.random.normal(0.0, 0.91, 1)
    x0 = np.array([v0, w0]).flatten()
    # small random perturbation to initial condition
    # setting arrays
    n_steps = int(tmax/dt)
    t = np.arange(0,tmax, dt)
    x = np.zeros((n_steps, len(x0)))

    # initial condition
    x[0] = x0

    for i in range(1, n_steps):

        xt = x[i-1]
        qtqdt = (dt) * (fitzhugh_nagumo(xt, args))
        #noise = noise_strength * np.sqrt(dt) * np.random.randn()

        x[i] = xt + qtqdt
    
    # Add noise after the integration only to the v variable
    noise = np.random.normal(0, noise_strength, size=x[:,0].shape)
    x[:,0] += noise
    return t, x[:,0], x[:,1]

def generate_fhn_obs_signal_once(length, dt, x0, args):
    """
    Generate a single time series signal using the FitzHugh-Nagumo model.

    Parameters:
    - length (int): Number of time steps to generate.
    - dt (float): Time step size.
    - x0 (list or np.ndarray): Initial state [v, w].
    - args (list): Parameters [b0, b1, epsilon, I, noise_strength].

    Returns:
    - t (np.ndarray): Time vector.
    - v (np.ndarray): Membrane potential.
    - w (np.ndarray): Recovery variable.
    """
    tmax = length * dt
    t, v, _ = SDEs_fhn(x0, tmax, dt, args)
    return t, v