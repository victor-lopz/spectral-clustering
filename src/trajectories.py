import numpy as np
import scipy.integrate
import scipy.spatial.distance
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
    - condicions_inicials: llista que conté condicions inicials [[x0,y0], [x1,y1],...]
    - t_span: tupla (t_inici, t_final) que indica l'interval de temps a simular
    - t_valors: np.array[float], conté els instants de temps on avaluem l'EDO
    - dimensio: dimensió dels punts a l'espai R^n (per defecte es 2 a R^2)
    
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
    Per això multipliquem per (t_steps - 1) al final.
    """
    num_trajectories, t_steps, dimensio = trajectories.shape
    distancies_1d = 0.5 * scipy.spatial.distance.pdist(trajectories[:, 0, :])
    for t in range(1, t_steps - 1):
        distancies_1d += scipy.spatial.distance.pdist(trajectories[:, t, :])
    distancies_1d += 0.5 * scipy.spatial.distance.pdist(trajectories[:, -1, :])
    pesos_1d = (t_steps - 1) / distancies_1d
    # Converteix al format de matriu simètrica amb diagonal zero.
    matriu_pesos = scipy.spatial.distance.squareform(pesos_1d)
    return matriu_pesos