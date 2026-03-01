import numpy as np
import scipy
from typing import Tuple, Dict
from sklearn.cluster import KMeans

def calcula_matriu_pesos(trajectories: np.ndarray, t_valors: np.ndarray) -> np.ndarray:
    num_trajectories = trajectories.shape[0]
    t_span_size = t_valors[-1] - t_valors[0]
    matriu_pesos = np.zeros((num_trajectories, num_trajectories))
    for i in range(num_trajectories - 1):
        # Calcula la diferència de la trajectòria 'i' amb totes les 'j > i' alhora
        # trajectories[i] té forma (T, 2), trajectories[i+1:] té forma (N-i-1, T, 2)
        diff = trajectories[i] - trajectories[i+1:]
        # Calcula la norma euclidiana per a cada punt temporal: forma (N-i-1, T)
        norms = np.linalg.norm(diff, axis=2)
        # Integra les distàncies usant la regla del trapezi: forma (N-i-1,)
        distancies = scipy.integrate.trapezoid(norms, x=t_valors, axis=1) / t_span_size
        # Calcula el pes i omple la matriu simètricament
        pesos = 1.0 / distancies
        matriu_pesos[i, i+1:] = pesos
        matriu_pesos[i+1:, i] = pesos
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
    diffs = np.diff(vaps)
    num_clusters = np.argmax(diffs)
    # sumem 2 perquè abans hem tret el VAP zero i per 
    # inloure el cluster dels estats incoherents
    return int(num_clusters) + 2

def troba_clusters(num_clusters: int, veps: np.ndarray) -> np.ndarray:
    matriu_veps_U = veps[:, :num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=7)
    labels = kmeans.fit_predict(matriu_veps_U)
    return labels