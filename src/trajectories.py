from typing import Callable

import numpy as np
import scipy.integrate
from scipy.spatial.distance import pdist, squareform

from src.datatypes import SpectralClusteringConfig


def generate_initial_conditions(params: SpectralClusteringConfig) -> np.ndarray:
    """
    Creates a rectangular grid of points in R^2 bounded by the limits defined by
    params.x_min, params.x_max, params.y_min, params.y_max.
    The spacing between points is given by params.grid_spacing.

    Returns a matrix of size (num_points, 2) where each row is a point (x,y).
    num_points is calculated to cover the entire space with the indicated spacing.

    Examle output: [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    """
    num_x = int(round((params.x_max - params.x_min) / params.grid_spacing)) + 1
    num_y = int(round((params.y_max - params.y_min) / params.grid_spacing)) + 1
    x = np.linspace(params.x_min, params.x_max, num_x)
    y = np.linspace(params.y_min, params.y_max, num_y)
    grid = np.empty((num_x * num_y, 2))
    grid[:, 0] = np.repeat(x, num_y)  # repeat every element of x num_y times
    grid[:, 1] = np.tile(y, num_x)  # repeat every element of y num_x times
    return grid


def simulate_trajectories(
    ode: Callable, initial_conditions: np.ndarray, params: SpectralClusteringConfig
) -> np.ndarray:
    """
    Parameters:
    - ode: Callable function that takes (t, z) and returns the vector field of the ODE,
        where t is the time and z is the position.
    - initial_conditions: list [[x0,y0], [x1,y1],...]
    - params: SpectralClusteringConfig object with the constants of the simulation:
        - t_span: tuple[float, float], with the time interval to simulate (start, end).
        - t_valors: np.array[float], contains the time instants where we evaluate the ODE
        - num_dimensions: dimension of the points in the space R^n (default is 2).

    Returns:
        A 3D matrix of shape (num_trajectories, t_steps, num_dimensions),
        on each trajectory is the solution of the ODE evaluated at the
        time instants indicated by t_valors.
    """
    num_trajectories = len(initial_conditions)
    y0_flat = initial_conditions.T.flatten()  # [x0,x1,x2,..., y0,y1,y2,...]

    def ode_vectorial(t, y_flat):
        # Transform the 1D vector to a matrix of size (num_dimensions, num_trajectories)
        z = y_flat.reshape(params.num_dimensions, num_trajectories)
        # Evaluate the ODE for each trajectory and return a flattened vector
        return np.array(ode(t, z)).flatten()

    sol = scipy.integrate.solve_ivp(
        ode_vectorial, params.t_span, y0_flat, t_eval=params.t_values
    )
    if sol.status != 0:
        raise RuntimeError(f"solve_ivp error: {sol.message}")

    t_steps = len(params.t_values)
    # sol.y has shape (num_dimensions * num_trajectories, t_steps)
    y = sol.y.reshape(params.num_dimensions, num_trajectories, t_steps)
    # permute the axes to go from (num_dimensions, num_trajectories, t_steps) to
    # (num_trajectories, t_steps, num_dimensions)
    trajectories = y.transpose(1, 2, 0)
    return trajectories


def calcula_matriu_pesos(trajectories: np.ndarray) -> np.ndarray:
    """
    Returns the weight matrix, where the weight between two trajectories is the
    inverse of their average distance over time.

    Procedure: at a fixed time instant t, the pdist function calculates all
    N*(N-1)/2 pairwise Euclidean distances between the N trajectories. We then use the
    trapezoidal rule to integrate over time: summing from k=1 to k=T-2 with weight
    1, and adding k=0 and k=T-1 with weight 1/2. Note that T = t_steps = len(t_values).

    Since the time step of t_values is uniform (np.linspace), the trapezoidal
    rule reduces to:
    r_ij ≈ [d_0/2 + d_1 + ... + d_{T-2} + d_{T-1}/2] / (T-1)

    Each weight w_ij is the inverse of the distance r_ij, which is why we
    multiply by (t_steps - 1) at the end.

    To avoid confusion with the already sparsified affinity matrix, we call
    this matrix the "weight matrix" instead of the "affinity matrix".
    """
    _num_trajectories, t_steps, _num_dimensions = trajectories.shape
    # Get pairwise Euclidean distances between all trajectories at each time instant.
    distance_vector = 0.5 * pdist(trajectories[:, 0, :])
    for t in range(1, t_steps - 1):
        distance_vector += pdist(trajectories[:, t, :])
    distance_vector += 0.5 * pdist(trajectories[:, -1, :])
    weight_vector = (t_steps - 1) / distance_vector
    # convert the list of weights to a symmetric matrix with zero diagonal.
    weight_matrix = squareform(weight_vector)
    return weight_matrix
