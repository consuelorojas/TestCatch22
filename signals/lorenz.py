import numpy as np

'''
Generate a Lorenz attractor that works with the distpacher framework.
'''

def _lorenz_equations(state, sigma, rho, beta):
    '''
    Compute the deterministic Lorenz system derivates.

    Parameters:
    - state (list or np.ndarray): Current state [x, y, z].
    - sigma: Prandtl number.
    - rho: Rayleigh number.
    - beta: Physical dimension parameter.

    Returns:
    - dvec (np.ndarray): Derivatives [dx/dt, dy/dt, dz/dt].
    '''

    x, y, z = state
    dx = sigma * (y-x)
    dy = x * (rho - z) - y
    dz = x*y - beta*z

    return np.array([dx, dy, dz])

def lorenz_base(args):
    '''
    Generate a Lorenz attractor trajectory with optional Gaussian noise.

    Parameters:
    - args (list or tuple): [[x0, y0, z0], sigma, rho, beta, dt, steps, noise_strength]
        - [x0, y0, z0]: Initial state.
        - sigma: Prandtl number.
        - rho: Rayleigh number.
        - beta: Physical dimension parameter.
        - dt: Time step size.
        - steps: Number of time steps to simulate.
        - noise_strength: Standard deviation of Gaussian noise (0 for no noise).
    '''

    state0, sigma, rho, beta, dt, steps, noise_strength = args

    # initial state
    state = np.array(state0, dtype=float)
    t = 0.0

    # storage
    traj = np.zeros((steps, 3))
    times = np.zeros(steps)

    for i in range(steps):
        dvec = _lorenz_equations(state, sigma, rho, beta)

        # gaussian noise
        noise = np.random.normal(0, 1, size=3) * noise_strength
        dvec = dvec +  noise

        # euler integration
        state+=dt*dvec
        t+= dt

        traj[i] = state
        times[i] = t

    return times, traj

def lorenz_base_1d(args, coord='x'):
    '''
    Retrieve only one dimension of the trajectories. By default retrieves x coordenate.
    '''
    times, traj = lorenz_base(args)
    idx = {'x':0, 'y':1, 'z':2}[coord]
    return times, traj[:, idx]


def generate_lorenz_once(args, coord=None):
    """
    Generate a single time series using the lorenz attractor equations.

    Parameters:
    - args (list): [sigman, rho, beta, dt, steps, noise_strengt]
    - coord (str or None): If None, return full 3D trajectory.
                            If 'x, 'y', or 'z', returns only that coordinate.

    Returns:
    - t (np, ndarray): Time Vector.
    - traj (np.ndarray): array size (steps, 3) if full trajectory or
                            or (steps,) for single coordinate.

    """

    if coord is None:
        t, traj = lorenz_base(args)
        return t, traj
    
    else:
        t, series = lorenz_base_1d(args, coord=coord)
        return t, series