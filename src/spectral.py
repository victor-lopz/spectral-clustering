import numpy as np
import scipy
from typing import Tuple, Dict
from sklearn.cluster import KMeans

def calcula_matriu_pesos(trajectories: np.ndarray) -> np.ndarray:
    num_trajectories, t_steps, dimensio = trajectories.shape
    # t_valors és sempre uniforme (np.linspace), de manera que la regla del trapezi
    # es redueix a:  r_ij ≈ [d_0/2 + d_1 + ... + d_{T-2} + d_{T-1}/2] / (T-1)
    # pdist calcula totes les N*(N-1)/2 distàncies euclidianes en un instant t.
    # Els extrems del trapezi compten la meitat: 
    # sumem des de k=1 fins a T-2 amb pes 1, i afegim k=0 i k=T-1 amb pes 1/2
    distancies_1d = 0.5 * scipy.spatial.distance.pdist(trajectories[:, 0, :])
    for t in range(1, t_steps - 1):
        distancies_1d += scipy.spatial.distance.pdist(trajectories[:, t, :])
    distancies_1d += 0.5 * scipy.spatial.distance.pdist(trajectories[:, -1, :])
    # Converteix al format de matriu simètrica amb diagonal zero.
    matriu_pesos = scipy.spatial.distance.squareform((t_steps - 1) / distancies_1d)
    return matriu_pesos

def calcula_estadistics(matriu_pesos: np.ndarray) -> Dict[str, float]:
    triangular_upper = matriu_pesos[np.triu_indices(len(matriu_pesos), k=1)]
    percentils = np.percentile(triangular_upper, [0, 50, 90, 95, 100])
    pes_min, pes_mediana, percentil90, percentil95, pes_max = percentils
    estadistics = {
        "pes_min":     pes_min,
        "pes_mediana": pes_mediana,
        "pes_mitja":   np.mean(triangular_upper),
        "percentil90": percentil90,
        "percentil95": percentil95,
        "pes_max":     pes_max
    }
    return estadistics

def imprimeix_estadistics(matriu_pesos: np.ndarray) -> None:
    print(f"Dimensions de la matriu de pesos: {matriu_pesos.shape}")
    estadistics = calcula_estadistics(matriu_pesos)
    for nom, valor in estadistics.items():
        print(f"{nom:<12} {valor:.3f}")

def sparcify_with_tol(matriu: np.ndarray, tol: float) -> Tuple[np.ndarray, float]:
    """Retorna una matriu on els elements més petits que la tolerància es tornen zero.
    Retorna també el percentatge d'esparsificació obtingut.
    Requisit: la diagonal de la matriu ha de ser zero."""
    sota_tolerancia = matriu < tol
    matriu_esparsa = np.where(sota_tolerancia, 0, matriu)
    zeros = np.sum(sota_tolerancia) - len(matriu)  # Excloem la diagonal
    total_elements = matriu.size - len(matriu)  # Excloem la diagonal
    percentatge_esparsificacio = float(zeros / total_elements)
    return matriu_esparsa, percentatge_esparsificacio

def calcula_tol_esparsificacio(matriu: np.ndarray, percent: float) -> float:
    """La tolerància o radi d'esparsificació és el percentil {percent}
    dels valors no nuls de la {matriu}. 
    Com la matriu és simètrica i la diagonal només conté zeros, 
    ens quedem amb la matriu triangular superior sense la diagonal
    """
    triangular_upper = matriu[np.triu_indices(len(matriu), k=1)]
    return float(np.percentile(triangular_upper, percent))

def sparcify(matriu: np.ndarray, percent: float) -> Tuple[np.ndarray, float, float]:
    """Retorna una matriu esparsa on el percentatge escollit dels 
    elements més petits es tornen zero.
    Requisit: la matriu ha de ser simètrica amb diagonal nul·la."""    
    tol = calcula_tol_esparsificacio(matriu, percent)
    matriu_esparsa, sparsification_percent = sparcify_with_tol(matriu, tol)
    return matriu_esparsa, tol, sparsification_percent

def calcula_matriu_grau(matriu_similaritat: np.ndarray) -> np.ndarray:
    return np.diag(matriu_similaritat.sum(axis=1))

def calcula_vaps_veps(matriu_grau_D:        np.ndarray, 
                      matriu_similaritat_W: np.ndarray, 
                      max_clusters:         int) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna els VAPs més petits ordenats ascendentment i
    els VEPs del problema generalitzat Lu = lambda Du.
    El paràmetre max_clusters fixa quants VAPs es calculen."""
    matriu_laplacia_L = matriu_grau_D - matriu_similaritat_W
    vaps, veps = scipy.linalg.eigh(matriu_laplacia_L, matriu_grau_D, 
                                   subset_by_index=[0, max_clusters])
    return vaps, veps

def calcula_num_clusters(vaps: np.ndarray) -> int:
    """Retorna el nombre de clusters a partir dels VAPs, segons la regla del colze.
    Aquesta regla diu que el nombre de clusters és el valor de k on la diferència
    entre vaps[k] i vaps[k-1] és màxima."""
    vaps_positius = vaps[1:] # eliminem el 1r VAP perquè és zero
    diffs = np.diff(vaps_positius)
    num_clusters = np.argmax(diffs)
    # sumem 2 perquè abans hem tret el VAP zero i per 
    # inloure el cluster dels estats incoherents
    return int(num_clusters) + 2

def troba_clusters(num_clusters: int, veps: np.ndarray) -> np.ndarray:
    matriu_veps_U = veps[:, :num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=7)
    labels = kmeans.fit_predict(matriu_veps_U)
    return labels