import numpy as np
import scipy
from typing import Tuple, Callable

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

def generar_trajectories(
    edo: Callable,
    condicions_inicials: np.ndarray,
    t_span: Tuple[float, float],
    t_valors: np.ndarray,
    dimensio: int = 2
) -> np.ndarray:
    """
    Paràmetres:
    - edo: funció que representa el camp vectorial d'una EDO
    - condicions_inicials: llista que conté condicions inicials [x0,y0]
    - t_span: tupla (t_inici, t_final) que indica l'interval de temps a simular
    - t_valors: np.array[float], conté els instants de temps on avaluem l'EDO
    - dimensio: dimensió dels punts a l'espai R^n (per defecte R^2)
    
    Retorna: 
        matriu 3D de mida (num_trajectories, t_steps, dimensio)
        on cada trajectòria és la solució de l'EDO avaluada en els 
        instants de temps indicats per t_valors.
    """
    num_trajectories = len(condicions_inicials)
    t_steps = len(t_valors)
    y0_flat = condicions_inicials.T.flatten()  # [x0,x1,x2,..., y0,y1,y2,...]

    def edo_vectorial(t, y_flat):
        # Transformem vector 1D a matriu de mida (dimensio, num_trajectories)
        z = y_flat.reshape(dimensio, num_trajectories)
        # Avaluem totes les trajectories alhora i tornem a aplanar
        return np.array(edo(t, z)).flatten()
        
    sol = scipy.integrate.solve_ivp(edo_vectorial, t_span, y0_flat, t_eval=t_valors)
    if sol.status != 0:
        raise RuntimeError(f"solve_ivp error: {sol.message}")
    
    # sol.y té mida (dimensio * num_trajectories, t_steps)
    # reconstruïm les dimensions
    y = sol.y.reshape(dimensio, num_trajectories, t_steps)
    # reordenem els eixos: de (dimensio, num_trajectories, t_steps) 
    # a (num_trajectories, t_steps, dimensio)
    trajectories = y.transpose(1, 2, 0)
    return trajectories