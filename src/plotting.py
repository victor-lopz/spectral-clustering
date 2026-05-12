import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Iterable
from datetime import datetime
import os

    
def grafica_trajectories(trajectories: np.ndarray,
                         desa_pdf: bool = False
                         ) -> None:
    for trajectoria in trajectories:
        coordenades_x = trajectoria[:,0]
        coordenades_y = trajectoria[:,1]
        plt.plot(coordenades_x, coordenades_y)
    for trajectoria in trajectories:
        coordenades_x = trajectoria[:,0]
        coordenades_y = trajectoria[:,1]
        pos_inicial = (coordenades_x[0], coordenades_y[0])
        inici_string = f'Inici = ({pos_inicial[0]:.2f}, {pos_inicial[1]:.2f})'
        plt.plot(pos_inicial[0], pos_inicial[1], 'o', label=inici_string, color='grey', markersize=4)
        pos_final = (coordenades_x[-1], coordenades_y[-1])
        final_string = f'Final = ({pos_final[0]:.2f}, {pos_final[1]:.2f})'
        plt.plot(pos_final[0], pos_final[1], 'o', label=final_string, color='red', markersize=4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    if desa_pdf: plt.savefig('edo.pdf')
    plt.show()
    
    
def grafica_punts(punts: Iterable) -> None:
    """punts: conjunt de punts a R^2. Exemple: [[0,1], [0.5,1], [1,1]]"""
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
    
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    
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


def set_custom_xtick(y_vals: np.ndarray, at_index: int) -> None:
    ax = plt.gca()
    default_ticks = ax.get_xticks()
    default_ticks = default_ticks[(default_ticks >= 0) & (default_ticks <= y_vals.size)]
    all_ticks = np.unique(np.append(default_ticks, at_index))
    ax.set_xticks(all_ticks)
    for tick_value, tick_label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if np.isclose(tick_value, at_index):
            tick_label.set_color('tab:red')


def grafica_eigenvalues_vs_index(eigenvalues: np.ndarray, output_dir: str = "../output/") -> None:
    """
    Grafica els valors propis ordenats de menor a major versus el seu index natural.
    També destaca el major eigengap per identificar visualment el nombre de clusters.
    """
    vals = np.sort(np.asarray(eigenvalues).ravel())
    if vals.size < 2:
        raise ValueError("Calen com a minim 2 valors propis per calcular l'eigengap.")

    indexes = np.arange(1, vals.size + 1)
    gaps = np.diff(vals)
    max_gap_pos = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_pos])
    k = max_gap_pos + 1

    plt.figure(figsize=(9, 5))
    plt.plot(indexes, vals, marker='o', linestyle='-', color='tab:blue', label='Valors propis')
    plt.axvline(k, color='tab:red', linestyle='--', alpha=0.8, label=f'Max eigengap en k={k}')
    plt.axvline(k + 1, color='tab:red', linestyle='--', alpha=0.5)
    plt.plot([k, k + 1], [vals[k - 1], vals[k]], color='tab:red', linewidth=2.5)
    set_custom_xtick(vals, at_index=k)
    plt.annotate(
        f'Max gap = {max_gap:.1e}',
        xy=(k + 0.5, 0.5 * (vals[k - 1] + vals[k])),
        xytext=(k + 4, 0.4 * (vals[k - 1] + vals[k])),
        arrowprops=dict(arrowstyle='->', color='tab:red'),
        fontsize=12,
        color='tab:red'
    )

    plt.xlabel(r'Índex $k$')
    plt.ylabel(r'Valor propi ($\lambda_{k}$)')
    plt.title('Valors propis respecte al seu índex')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    time = datetime.now().strftime('%H-%M-%S')
    filename = f"{time}_eigenvalues_vs_index.pdf"
    date = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, date)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    plt.show()


def grafica_eigengaps_vs_index(eigenvalues: np.ndarray, output_dir: str = "../output/") -> None:
    """
    Grafica els eigengaps (diferències consecutives de valors propis ordenats)
    respecte al seu index natural i destaca el màxim eigengap.
    """
    vals = np.sort(np.asarray(eigenvalues).ravel())
    if vals.size < 2:
        raise ValueError("Calen com a minim 2 valors propis per calcular els eigengaps.")

    gaps = np.diff(vals)
    indexes = np.arange(1, gaps.size + 1)
    max_gap_pos = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_pos])
    k = max_gap_pos + 1

    plt.figure(figsize=(9, 5))
    plt.plot(indexes, gaps, marker='o', linestyle='-', color='tab:blue', label='Eigengaps')
    plt.axvline(k, color='tab:red', linestyle='--', alpha=0.8, label=f'Max eigengap en k={k}')
    plt.scatter([k], [max_gap], color='tab:red', zorder=3)
    set_custom_xtick(gaps, at_index=k)
    plt.annotate(
        f'Max gap = {max_gap:.1e}',
        xy=(k, max_gap),
        xytext=(k + 3, 0.95 * max_gap),
        arrowprops=dict(arrowstyle='->', color='tab:red'),
        fontsize=12,
        color='tab:red'
    )

    plt.xlabel(r'Índex $k$')
    plt.ylabel(r'Eigengap ($\lambda_{k+1} - \lambda_{k}$)')
    plt.title('Eigengaps respecte al seu índex')
    # plt.xticks(indexes)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    time = datetime.now().strftime('%H-%M-%S')
    filename = f"{time}_eigengaps_vs_index.pdf"
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
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    descripcio = (
        f"{num_clusters} clusters, {num_trajectories} trajectòries, "
        f"{t_steps} passes de temps," "\n"
        f"temps final = {t_span[-1]:.1f}s, "
        f"esparsificació = {percent_esparsificacio*100:.0f}%"
        f", radi = {radi_esparsificacio:.2f}"
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
        f"_sparse={percent_esparsificacio*100:.0f}"
        ".pdf"
    )
    date = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, date)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    plt.show()