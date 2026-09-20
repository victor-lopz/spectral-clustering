import numpy as np


def ode_autonomous_duffing(t: np.ndarray, z: np.ndarray) -> list[np.ndarray]:
    """Parameters:
        t: time
        z: position in the R^2 plane
    Returns the vector field of the autonomous Duffing system x'=y; y'=x-x^3.
    """
    x, y = z
    return [y, x - x**3]


def ode_non_autonomous_duffing(t: np.ndarray, z: np.ndarray) -> list[np.ndarray]:
    """Parameters:
        t: time
        z: position in the R^2 plane
    Returns the vector field of the non-autonomous Duffing system
    x_dot = - dPsi/dy; y_dot = dPsi/dx, where Psi is the stream function
    of the non-autonomous Duffing system.
    """
    x, y = z
    a = 0.5  # Non-linear coefficient
    delta_0 = 0.1  # Pertorbation amplitude
    theta = 3 * np.pi / 2  # Angular frequency
    phi = np.pi / 4  # Initial phase
    cosine = np.cos(theta * t + phi)

    def delta(x):
        return delta_0 * (x + 4.5) / 2

    def delta_derivative(x):
        return delta_0 / 2

    x_dot = y
    first = -(delta_derivative(x) * cosine) * (x**2 / 2 - a * x**4 / 4)
    second = (1 - delta(x) * cosine) * (x - a * x**3)
    y_dot = first + second
    return [x_dot, y_dot]
