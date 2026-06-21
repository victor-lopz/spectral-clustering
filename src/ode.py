import numpy as np


def edo_duffing_autonom(t, z):
    """Paràmetres:
        t: temps (no s'utilitza)
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
    x_dot = - dPsi/dy; y_dot = dPsi/dx, on Psi és la funció de corrent
    de Duffing no autònom.
    """
    x, y = z
    a = 0.5  # Coeficient de no linealitat
    delta_0 = 0.1  # Amplitud base de la pertorbació
    theta = 3 * np.pi / 2  # Freqüència angular
    phi = np.pi / 4  # Fase inicial
    cosinus = np.cos(theta * t + phi)

    def delta(x):
        return delta_0 * (x + 4.5) / 2

    def delta_derivada(x):
        return delta_0 / 2

    x_dot = y
    primer = -(delta_derivada(x) * cosinus) * (x**2 / 2 - a * x**4 / 4)
    segon = (1 - delta(x) * cosinus) * (x - a * x**3)
    y_dot = primer + segon
    return [x_dot, y_dot]
