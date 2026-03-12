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


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


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


def grafica_eigengaps_vs_radi(diffs_max: list[float], 
                              nums_clusters: list[int], 
                              radis: np.ndarray,
                              estadistics: dict[str, float],
                              sparsificacions: list[float],
                              output_dir: str = "../output/"
                              ) -> None:
    """
    Grafica el nombre de clusters, l'eigen gap i el percentatge d'esparsificació
    en funció del radi d'esparsificació.
    """
    fig, host = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", 1.1))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)
    color_gap = 'tab:blue'
    color_clust = 'tab:red'
    color_sparse = 'tab:green'
    p1, = host.plot(radis, diffs_max, marker='.', color=color_gap, label='Eigen gap')
    p2, = par1.plot(radis, nums_clusters, marker='.', color=color_clust, label='Nombre de clusters')
    p3, = par2.plot(radis, sparsificacions, marker='.', color=color_sparse, label='Esparsificació (%)')
    host.set_xlabel("Radi d'esparsificació")
    host.set_ylabel('Diferència màxima entre VAPs consecutius', color=color_gap)
    par1.set_ylabel('Nombre de clusters', color=color_clust)
    par2.set_ylabel('Esparsificació (%)', color=color_sparse)
    host.tick_params(axis='y', labelcolor=color_gap)
    par1.tick_params(axis='y', labelcolor=color_clust)
    par2.tick_params(axis='y', labelcolor=color_sparse)
    host.yaxis.get_offset_text().set_horizontalalignment('left')
    host.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

    colors_stats = iter(plt.rcParams['axes.prop_cycle'])
    for nom, valor in estadistics.items():
        if nom != "pes_max":
            host.axvline(x=valor, linestyle='--', alpha=0.6,
                        color=next(colors_stats)['color'],
                        label=f'{nom} = {valor:.2f}')

    host.set_title("Evolució dels indicadors vs Radi d'esparsificació")
    host.grid(True, alpha=0.3)
    lines = [p1, p2, p3]
    labels: list[str] = [str(l.get_label()) for l in lines]
    for l in host.get_lines():
        if l not in lines:
            lines.append(l)
            labels.append(str(l.get_label()))
    host.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    time = datetime.now().strftime('%H-%M-%S')
    filename = f"{time}_eigengap_vs_radi.pdf"
    date = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, date)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    plt.show()


def grafica_clusters(condicions_inicials: np.ndarray, 
                     labels: np.ndarray, 
                     num_clusters: int, 
                     radi_esparsificacio: float, 
                     percent_esparsificacio: float, 
                     t_steps: int, 
                     t_span: Tuple[float, float],
                     output_dir: str = "../output/") -> None:
    
    num_trajectories = len(condicions_inicials)
    for cluster_id in range(num_clusters):
        indices = np.where(labels == cluster_id)
        if len(indices[0]) > 0:
            plt.scatter(condicions_inicials[indices, 0], 
                        condicions_inicials[indices, 1], 
                        s=40, label=cluster_id)
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
        f"temps final = {t_span[-1]:.1f}s, esparsificació = {percent_esparsificacio*100:.0f}%"
    )
    plt.figtext(0.5, 0.01, descripcio, ha='center', fontsize=11)
    plt.subplots_adjust(bottom=0.1)
    time = datetime.now().strftime('%H-%M-%S')
    filename = (
        f"{time}"
        f"_clusters={num_clusters}"
        f"_traj={num_trajectories}"
        f"_tsteps={t_steps}"
        f"_t_end={t_span[-1]:.1f}"
        f"_tol={radi_esparsificacio:.1f}"
        f"_sparse={percent_esparsificacio*100:.0f}%"
        ".pdf"
    )
    date = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, date)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename))
    plt.show()