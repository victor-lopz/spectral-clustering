import numpy as np
import scipy.linalg
from typing import Tuple, Dict
from sklearn.cluster import KMeans

from src.datatypes import ParametresGenerals, SpectralAnalysisResult
from src.plotting import grafica_clusters


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
    Opcionalment, retorna també el percentatge d'esparsificació obtingut.
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
    radi_esparsificacio = np.percentile(triangular_upper, percent)
    return float(radi_esparsificacio)


def sparcify(matriu: np.ndarray, percent: float) -> Tuple[np.ndarray, float, float]:
    """Retorna una matriu esparsa on el percentatge escollit dels 
    elements més petits es tornen zero.
    Requisit: la matriu ha de ser simètrica amb diagonal nul·la."""
    tol = calcula_tol_esparsificacio(matriu, percent)
    matriu_esparsa, sparsification_percent = sparcify_with_tol(matriu, tol)
    return matriu_esparsa, tol, sparsification_percent


def calcula_matriu_grau(matriu_similaritat: np.ndarray) -> np.ndarray:
    """Calcula la suma de cada fila i les col·loca en una matriu diagonal."""
    return np.diag(matriu_similaritat.sum(axis=1))


def calcula_vaps(matriu_similaritat_W: np.ndarray, 
                 max_clusters: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna els n VAPs més petits ordenats ascendentment i
    els VEPs del problema generalitzat Lu = lambda Du.
    Requisit: les matrius L i D han de ser simètriques."""
    n = matriu_similaritat_W.shape[0]
    if n == 0:
        raise ValueError("La matriu de similaritat ha de tenir almenys una trajectòria.")
    if max_clusters <= 0:
        raise ValueError(f"cal max_clusters > 0, rebut: {max_clusters}.")
    max_index = min(max_clusters - 1, n - 1)
    matriu_grau_D = calcula_matriu_grau(matriu_similaritat_W)
    matriu_laplacia_L = matriu_grau_D - matriu_similaritat_W
    vaps, veps = scipy.linalg.eigh(matriu_laplacia_L, matriu_grau_D, 
                                   subset_by_index=[0, max_index])
    return vaps, veps


def calcula_num_clusters_i_max_eigengap(vaps: np.ndarray) -> Tuple[int, float]:
    """Retorna el nombre de clusters segons la regla del colze.
    Aquesta regla diu que el nombre de clusters és el valor de l'índex k 
    on la diferència entre vaps[k] i vaps[k-1] és màxima. És a dir, és 
    l'argument del màxim de diferències consecutives de VAPs ordenats.
    També retorna el valor de la diferència màxima trobada (max eigengap).

    Es suma 1 perquè els VAPs es compten des de l'índex zero i, a la fórmula
    de l'article, es compten des de l'índex 1. A més, es suma 1 més per
    incloure el cluster dels estats incoherents.
    """
    diffs = np.diff(vaps)
    k = int(np.argmax(diffs))
    diff_max = diffs[k]
    num_clusters = k + 2
    return num_clusters, diff_max


def troba_clusters(num_clusters: int, veps: np.ndarray) -> np.ndarray:
    """
    Retorna un vector d'etiquetes de clusters per a cada trajectòria.
    Exemple: labels[i] = 0 indica que la trajectòria i pertany al cluster 0.
    """
    matriu_veps_U = veps[:, :num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=7)
    labels = kmeans.fit_predict(matriu_veps_U)
    return labels


def calcula_indicadors_vs_radis(matriu_pesos: np.ndarray,
                                constant_diagonal: float,
                                params: ParametresGenerals
                                ) -> SpectralAnalysisResult:
    """
    Retorna una classe SpectralAnalysisResult que, 
    per a cada radi d'esparsificació, conté:
    - la diferència màxima entre VAPs consecutius
    - la diferència màxima normalitzada entre VAPs consecutius
    - el nombre de clusters trobat, sempre dins del rang [1, max_clusters]
    - el percentatge d'esparsificació
    - els estadístics de la matriu de pesos (min, max, mediana, mitja, percentils90 i 95)
    - tots els VEPs associats a cada radi d'esparsificació
    """
    estadistics = calcula_estadistics(matriu_pesos)
    radis = np.linspace(estadistics["pes_min"], estadistics["percentil95"], params.num_radis)
    result = SpectralAnalysisResult(radis=radis, estadistics=estadistics)
    for radi in radis:
        matriu_similaritat_W, percent = sparcify_with_tol(matriu_pesos, radi)
        result.sparsificacions.append(percent)
        np.fill_diagonal(matriu_similaritat_W, constant_diagonal)
        vaps, veps = calcula_vaps(matriu_similaritat_W, params.max_clusters)
        num_clusters, max_eigengap = calcula_num_clusters_i_max_eigengap(vaps)
        result.nums_clusters.append(num_clusters)
        result.eigengaps.append(max_eigengap)
        rang_espectral = vaps[-1] - vaps[0]
        gap_normalitzat = max_eigengap / rang_espectral if rang_espectral > 0 else 0
        result.normalized_eigengaps.append(gap_normalitzat)
        result.veps_list.append(veps)
    return result


def troba_indexs_max_rel(diffs: list[float]) -> list[int]:
    """Troba els màxims relatius d'un vector i retorna els seus índexs.
    S'utilitza per trobar els màxims relatius de les diferències 
    entre VAPs consecutius en funció del radi d'esparsificació."""
    maxs_rels = []
    i = 1
    while i < len(diffs) - 1:
        if diffs[i-1] < diffs[i] > diffs[i+1]:
            maxs_rels.append(i)
            i += 2
        else: i += 1
    return maxs_rels


def grafica_clusters_maxs_rel(
    indexs_max_rel: list[int],
    result: SpectralAnalysisResult,
    condicions_inicials: np.ndarray,
    params: ParametresGenerals,
    subfolder: str|None = None
) -> None:
    """Dibuixa els clusters trobats per cada radi d'esparsificació 
    que generi un màxim relatiu de les diferències entre VAPs consecutius."""
    for num, index in enumerate(indexs_max_rel, start=1):
        radi = result.radis[index]
        percent = result.sparsificacions[index]
        n_clusters = result.nums_clusters[index]
        diff_max = result.eigengaps[index]
        veps = result.veps_list[index]
        print(f"Maxim_relatiu_num {num}\n"
              f"Radi: {radi:.3f}, Esparsificació: {percent:.2%}, "
              f"Clusters: {n_clusters}, Max Eigen gap: {diff_max:.5e}")
        labels = troba_clusters(n_clusters, veps)
        grafica_clusters(condicions_inicials, labels, n_clusters, radi, percent, 
                         params, subfolder, filename_prefix=f"max_rel-{num}_")
