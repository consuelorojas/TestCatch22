import numpy as np

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
    # setting arrays
    x0 = np.random.randn(len(x0)) * 0.1 + x0  # small random perturbation to initial condition
    n_steps = int(tmax/dt)
    t = np.arange(0,tmax, dt)
    x = np.zeros((n_steps, len(x0)))

    # initial condition
    x[0] = x0

    for i in range(1, n_steps):

        xt = x[i-1]
        qtqdt = (dt) * (fitzhugh_nagumo(x[i-1], args))
        noise = noise_strength * np.sqrt(dt) * np.random.randn()

        x[i] = xt + qtqdt + noise

    return t, x[:,0], x[:,1]

def generate_fhn_signal_once(length, dt, x0, args):
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