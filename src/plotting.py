import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Iterable
from datetime import datetime
import os

# Paràmetres
radi = 0.3
centre_esquerre = (-1, 0)
centre_dret = (1, 0)
centre_nord = (0, 1)
centre_sud = (0, -1)
resolucio_grafica = 2000

def grafica_circumferencia(centre: Tuple[float, float],
                           radi: float, 
                           nom_regio:str,
                           resolucio: int) -> None:
    theta = np.linspace(0, 2*np.pi, resolucio)
    a, b = centre
    xx = radi*np.cos(theta) + a
    yy = radi*np.sin(theta)
    plt.plot(xx, yy + b, 'black')
    plt.plot(xx, - yy + b, 'black')
    plt.annotate(nom_regio, centre)

def grafica_regio(nom_regio: str, radi: float, centres: list[Tuple[float, float]], resolucio: int
                  ) -> None:
    for centre in centres:
        grafica_circumferencia(centre, radi, nom_regio, resolucio)

def grafica_regions_A_B(radi: float, resolucio: int) -> None:
    grafica_regio('A', radi, [centre_esquerre, centre_dret], resolucio)
    grafica_regio('B', radi, [centre_nord, centre_sud], resolucio)
    
def grafica_trajectories(trajectories: np.ndarray,
                         dibuixa_regions: bool = False,
                         desa_pdf: bool = False, 
                         radi: float = radi, 
                         resolucio: int = resolucio_grafica
                         ) -> None:
    if dibuixa_regions: grafica_regions_A_B(radi, resolucio)
    for trajectoria in trajectories:
        coordenades_x = trajectoria[:,0]
        coordenades_y = trajectoria[:,1]
        plt.plot(coordenades_x, coordenades_y)
    for trajectoria in trajectories:
        coordenades_x = trajectoria[:,0]
        coordenades_y = trajectoria[:,1]
        pos_inicial = (coordenades_x[0], coordenades_y[0])
        inici_string = f'Inici = ({pos_inicial[0]:.2f}, {pos_inicial[1]:.2f})'
        plt.plot(pos_inicial[0], pos_inicial[1], '-o', label=inici_string, color='grey')
        pos_final = (coordenades_x[-1], coordenades_y[-1])
        final_string = f'Final = ({pos_final[0]:.2f}, {pos_final[1]:.2f})'
        plt.plot(pos_final[0], pos_final[1], '-o', label=final_string, color='red')
    plt.xlabel('x')
    plt.ylabel('y') 
    # plt.legend(); 
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    if desa_pdf: plt.savefig('edo.pdf')
    plt.show()
    
def grafica_punts(punts: Iterable, dibuixa_regions=False, radi=radi, resolucio=resolucio_grafica) -> None:
    """punts: conjunt de punts a R^2. Exemple: [[0,1], [0.5,1], [1,1]]"""
    if dibuixa_regions:
        grafica_regions_A_B(radi, resolucio)
    for punt in punts:
        plt.plot(punt[0], punt[1], 'o', markersize=5, color="grey")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Condicions inicials')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.show()

def grafica_dif_vs_radi(diffs_max: list[float], radis: np.ndarray) -> None:
    """Grafica la diferència màxima entre VAPs consecutius 
    en funció del radi d'esparsificació.
    """
    plt.plot(radis, diffs_max, marker='o')
    plt.xlabel("Radi d'esparsificació")
    plt.ylabel('Diferència màxima entre VAPs consecutius')
    plt.title("Diferència màxima entre VAPs vs Radi d'esparsificació")
    plt.grid()
    plt.show()

def grafica_clusters(condicions_inicials: np.ndarray, 
                     labels: np.ndarray, 
                     num_clusters: int, 
                     sparsification_tol: float, 
                     sparsification_percent: float, 
                     t_steps: int, 
                     t_span: Tuple[float, float],
                     output_dir: str = "../output/") -> None:
    num_trajectories = len(condicions_inicials)
    for cluster_id in range(num_clusters):
        indices = np.where(labels == cluster_id)
        if len(indices[0]) > 0:
            plt.scatter(condicions_inicials[indices, 0], 
                        condicions_inicials[indices, 1], 
                        s=30, label=cluster_id)
    plt.title('Clusters')
    plt.xlabel('x')
    plt.ylabel('y')
    max_legend_entries = 10
    if num_clusters < max_legend_entries:
        plt.legend(title="Cluster", loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    descripcio = (
        f"{num_clusters} clusters, {num_trajectories} trajectòries, "
        f"{t_steps} passes de temps," "\n"
        f"temps final = {t_span[-1]:.1f}s, esparsificació = {sparsification_percent*100:.0f}%"
    )
    plt.figtext(0.5, 0.01, descripcio, ha='center', fontsize=11)
    plt.subplots_adjust(bottom=0.1)
    timestamp = datetime.now().strftime('%H-%M-%S')
    filename = (
        f"{timestamp}"
        f"_clusters={num_clusters}"
        f"_traj={num_trajectories}"
        f"_tsteps={t_steps}"
        f"_t_end={t_span[-1]:.1f}"
        f"_tol={sparsification_tol:.1f}"
        f"_sparse={sparsification_percent*100:.0f}%"
        ".pdf"
    )
    date = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, date)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename))
    plt.show()