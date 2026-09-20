from typing import Callable

import numpy as np
import scipy.integrate
import scipy.spatial.distance

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


def generar_trajectories(
    edo: Callable, condicions_inicials: np.ndarray, params: SpectralClusteringConfig
) -> np.ndarray:
    """
    Paràmetres:
    - edo: funció que representa el camp vectorial d'una EDO
    - condicions_inicials: llista que conté condicions inicials [[x0,y0], [x1,y1],...]
    - params: objecte ParametresGenerals que conté les constants de la simulació:
        - t_span: tupla (t_inici, t_final) que indica l'interval de temps a simular
        - t_valors: np.array[float], conté els instants de temps on avaluem l'EDO
        - num_dimensions: dimensió dels punts a l'espai R^n (per defecte és 2 a R^2)

    Retorna:
        matriu 3D de mida (num_trajectories, t_steps, num_dimensions)
        on cada trajectòria és la solució de l'EDO avaluada en els
        instants de temps indicats per t_valors.
    """
    num_trajectories = len(condicions_inicials)
    y0_flat = condicions_inicials.T.flatten()  # [x0,x1,x2,..., y0,y1,y2,...]

    def edo_vectorial(t, y_flat):
        # Transformem vector 1D a matriu de mida (dimensio, num_trajectories)
        z = y_flat.reshape(params.num_dimensions, num_trajectories)
        # Avaluem totes les trajectories alhora i tornem a aplanar
        return np.array(edo(t, z)).flatten()

    sol = scipy.integrate.solve_ivp(
        edo_vectorial, params.t_span, y0_flat, t_eval=params.t_values
    )
    if sol.status != 0:
        raise RuntimeError(f"solve_ivp error: {sol.message}")

    t_steps = len(params.t_values)
    # sol.y té mida (dimensio * num_trajectories, t_steps)
    # reconstruïm les dimensions
    y = sol.y.reshape(params.num_dimensions, num_trajectories, t_steps)
    # permutem els eixos per pasar de (dimensio, num_trajectories, t_steps) a
    # (num_trajectories, t_steps, dimensio)
    trajectories = y.transpose(1, 2, 0)
    return trajectories


def calcula_matriu_pesos(trajectories: np.ndarray) -> np.ndarray:
    """
    Retorna la matriu de pesos on el pes entre dues trajectòries és
    la inversa de la distància mitjana al llarg del temps.

    Procediment: fixat un instant de temps t, la funció pdist calcula
    totes les N*(N-1)/2 distàncies euclidianes entre les N trajectòries.
    Després, usem la regla del trapezi per integrar al llarg del temps:
    sumem des de k=1 fins a T-2 amb pes 1, i afegim k=0 i k=T-1 amb pes 1/2.

    Com el pas de temps de t_valors és uniforme (np.linspace),
    la regla del trapezi es redueix a:
    r_ij ≈ [ d_0/2 + d_1 + ... + d_{T-2} + d_{T-1}/2 ] / (T-1)

    Cada pes w_ij és l'inversa de la distància r_ij,
    per això multipliquem per (t_steps - 1) al final.

    Per evitar confusió amb la matriu d'afinitat ja esparsificada, anomenem
    aquesta matriu "matriu_pesos" en comptes de "matriu_afinitat".
    """
    num_trajectories, t_steps, dimensio = trajectories.shape
    distancies_1d = 0.5 * scipy.spatial.distance.pdist(trajectories[:, 0, :])
    for t in range(1, t_steps - 1):
        distancies_1d += scipy.spatial.distance.pdist(trajectories[:, t, :])
    distancies_1d += 0.5 * scipy.spatial.distance.pdist(trajectories[:, -1, :])
    pesos_1d = (t_steps - 1) / distancies_1d
    # Converteix la llista de pesos al format de matriu simètrica amb diagonal zero.
    matriu_pesos = scipy.spatial.distance.squareform(pesos_1d)
    return matriu_pesos
