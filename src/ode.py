import numpy as np


def edo_duffing_autonom(t, z):
    """Paràmetres:
        t: temps
        z: posició al pla R^2
    Retorna el camp vectorial del sistema d'EDOs x'=y; y'=x-x^3.
    """
    x, y = z
    return [y, x - x**3]


def edo_duffing_no_autonom(t, z):
    """Paràmetres:
        t: temps
        z: posició al pla R^2
    Retorna el camp vectorial del sistema d'EDOs 
    x' = -dPsi/dy; y' = dPsi/dx, on Psi és la funció de flux de Duffing no autònom.
    """
    x, y = z
    a = 0.5
    eps = 0.1
    omega = 3 * np.pi / 2
    phi = np.pi / 4
    cosinus = np.cos(omega * t + phi)
    
    def epsilon(x):
        return eps * (x + 4.5) / 2
    
    def epsilon_dot(x):
        return eps / 2
  
    x_dot = y
    first = - (epsilon_dot(x) * cosinus) * (x**2 / 2 - a * x**4 / 4)
    second = (1 - epsilon(x) * cosinus) * (x - a * x**3)
    y_dot = first + second
    return [x_dot, y_dot]
