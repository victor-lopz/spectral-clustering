import numpy as np
import scipy
from typing import Tuple

def generar_condicions_inicials(
    step_size: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    verbose: bool = True
) -> np.ndarray:
    num_x = int(round((x_range[1] - x_range[0]) / step_size)) + 1
    num_y = int(round((y_range[1] - y_range[0]) / step_size)) + 1
    x = np.linspace(x_range[0], x_range[1], num_x)
    y = np.linspace(y_range[0], y_range[1], num_y)
    malla = np.empty((num_x * num_y, 2))
    malla[:, 0] = np.repeat(x, num_y) # repetim cada valor de x num_y vegades
    malla[:, 1] = np.tile(y, num_x) # repetim tot el vector y num_x vegades
    if verbose:
        print(f"Nombre de trajectòries = {len(malla)} = {num_y} files * {num_x} columnes")
    return malla

def generar_trajectories(edo,
                         condicions_inicials: np.ndarray,
                         t_span: Tuple[float, float],
                         t_valors: np.ndarray,
                         dimensio=2,
                         epsilon=0,
                         funcio_soroll=np.sin) -> np.ndarray:
    """
    Paràmetres:
        edo: funció que representa el camp vectorial d'una EDO
        condicions_inicials: llista que conté condicions inicials [x0,y0]
        t_valors: np.array[float], conté els instants de temps on avaluem l'EDO
    
    Cada trajectòria és la solució de l'EDO avaluada en els instants de temps indicats.
    """
    num_trajectories = len(condicions_inicials)
    t_steps = len(t_valors)
    y0_flat = condicions_inicials.flatten()

    def edo_vectorial(t, y_flat):
        # Transformem vector 1D a matriu (dimensio, num_trajectories)
        z = y_flat.reshape(num_trajectories, dimensio).T
        # Avaluem totes alhora i tornem a aplanar
        return np.array(edo(t, z)).T.flatten()
        
    sol = scipy.integrate.solve_ivp(edo_vectorial, t_span, 
                                    y0_flat, t_eval=t_valors)
    if sol.status != 0:
        raise RuntimeError(f"solve_ivp error: {sol.message}")
    # Reconstruim la matriu de trajectòries: 
    # de (num_trajectories * dimensio, t_steps) a (num_trajectories, t_steps, dimensio)
    trajectories = sol.y.T.reshape(t_steps, num_trajectories, dimensio).transpose(1, 0, 2)
    return trajectories