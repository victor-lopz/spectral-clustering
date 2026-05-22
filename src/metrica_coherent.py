"""
Mòdul per calcular la mètrica de coherència espectral.
Basat en l'article de Filippi et al., Figura 5.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def calcula_coherencia_cluster(W: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Calcula la mètrica de coherència per a un sol cluster específic.
    
    La coherència es defineix com la ràtio entre les connexions internes 
    del cluster (pes intern) i el volum total d'aquest cluster.
    """
    nodes_k = np.where(labels == k)[0]
    
    if len(nodes_k) == 0:
        return 0.0
        
    W_interna = W[np.ix_(nodes_k, nodes_k)]
    pes_intern = np.sum(W_interna)
    
    graus_nodes = np.sum(W[nodes_k, :], axis=1)
    volum_cluster = np.sum(graus_nodes)
    
    if volum_cluster == 0:
        return 0.0
        
    return pes_intern / volum_cluster


def grafica_coherencia(
    W: np.ndarray, 
    labels: np.ndarray, 
    pos_t0: np.ndarray, 
    pos_tf: Optional[np.ndarray] = None,
    titol: str = "Coherència dels clústers"
):
    """
    Genera un gràfic on cada partícula és acolorida segons la 
    coherència del seu cluster a l'espai 2D.
    
    Args:
        W: Matriu d'afinitat.
        labels: Les etiquetes d'agrupament calculades per un valor K determinat.
        pos_t0: Array de dimensions (N, 2) amb les posicions inicials (x, y).
        pos_tf: Array opcional de dimensions (N, 2) amb les posicions finals (x, y).
        titol: Títol de la figura.
    """
    
    ids_clusters = np.unique(labels)
    coherencia_map = {}
    for k in ids_clusters:
        coherencia_map[k] = calcula_coherencia_cluster(W, labels, k)
    
    coherencia_particules = np.array([coherencia_map[lbl] for lbl in labels])
    
    num_plots = 2 if pos_tf is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
        
    ax_t0 = axes[0]
    sc1 = ax_t0.scatter(
        pos_t0[:, 0], pos_t0[:, 1], 
        c=coherencia_particules, cmap='PuBuGn',
        s=10, alpha=0.9
    )
    ax_t0.set_title(r"$t_0$")
    ax_t0.set_xlabel("x")
    ax_t0.set_ylabel("y")
    ax_t0.set_aspect('equal', 'box')
    
    if pos_tf is not None:
        ax_tf = axes[1]
        sc2 = ax_tf.scatter(
            pos_tf[:, 0], pos_tf[:, 1], 
            c=coherencia_particules, cmap='PuBuGn', 
            s=10, alpha=0.9
        )
        ax_tf.set_title(r"$t_f$")
        ax_tf.set_xlabel("x")
        ax_tf.set_ylabel("y")
        ax_tf.set_aspect('equal', 'box')
        
    cbar = fig.colorbar(sc1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Coherència", rotation=90, labelpad=15)
    
    if titol:
        fig.suptitle(titol, fontsize=14, y=1.02)
        
    plt.show()