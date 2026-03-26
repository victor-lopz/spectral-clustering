import numpy as np
import scipy
from typing import Tuple, Dict
from sklearn.cluster import KMeans

from src.plotting import grafica_clusters


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
    Requisit: les dues matrius han de ser simètriques."""
    matriu_grau_D = calcula_matriu_grau(matriu_similaritat_W)
    matriu_laplacia_L = matriu_grau_D - matriu_similaritat_W
    vaps, veps = scipy.linalg.eigh(matriu_laplacia_L, matriu_grau_D, 
                                   subset_by_index=[0, max_clusters])
    return vaps, veps


def calcula_num_clusters_i_max_eigengap(vaps: np.ndarray) -> Tuple[int, float]:
    """Retorna el nombre de clusters segons la regla del colze.
    Aquesta regla diu que el nombre de clusters és el valor de l'índex k 
    on la diferència entre vaps[k] i vaps[k-1] és màxima. És a dir, és 
    l'argument del màxim de diferències consecutives de VAPs ordenats.
    Opcionalment, també retorna el valor de la diferència màxima trobada.
    """
    vaps_positius = vaps[1:] # eliminem el 1r VAP perquè és zero
    diffs = np.diff(vaps_positius)
    k = int(np.argmax(diffs))
    # sumem 2 perquè abans hem tret el VAP zero i per 
    # inloure el cluster dels estats incoherents
    num_clusters = k + 2
    diff_max = diffs[k]
    return num_clusters, diff_max


def troba_clusters(num_clusters: int, veps: np.ndarray) -> np.ndarray:
    """
    Retorna un vector d'etiquetes de clusters per a cada trajectòria.
    """
    matriu_veps_U = veps[:, :num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=7)
    labels = kmeans.fit_predict(matriu_veps_U)
    return labels


def calcula_diffs_vs_radis(matriu_pesos: np.ndarray, 
                           constant_diagonal: float,
                           condicions_inicials: np.ndarray,
                           t_steps: int, 
                           t_span: Tuple[float, float],
                           max_clusters: int = 20,
                           num_radis: int = 80
                           ) -> Tuple[list[float], list[int], np.ndarray, dict[str, float], list[float], list[np.ndarray]]:
    """
    Retorna les diferències màximes entre VAPs consecutius, el nombre de clusters 
    i el percentatge d'esparsificació en funció del radi d'esparsificació.
    """
    
    estadistics = calcula_estadistics(matriu_pesos)
    radis = np.linspace(estadistics["pes_min"], estadistics["percentil95"], num_radis)
    diffs_vaps = []
    nums_clusters = []
    sparsificacions = []
    veps_list = []
    for radi in radis:
        matriu_similaritat_W, percent = sparcify_with_tol(matriu_pesos, radi)
        sparsificacions.append(percent)
        np.fill_diagonal(matriu_similaritat_W, constant_diagonal)
        vaps, veps = calcula_vaps(matriu_similaritat_W, max_clusters)
        n_clusters, diff_max = calcula_num_clusters_i_max_eigengap(vaps)
        nums_clusters.append(n_clusters)
        diffs_vaps.append(diff_max)
        veps_list.append(veps)
    return diffs_vaps, nums_clusters, radis, estadistics, sparsificacions, veps_list


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


def grafica_clusters_maxs_rel(indexs_max_rel: list[int],
                              radis: np.ndarray,
                              sparsificacions: list[float],
                              nums_clusters: list[int],
                              diffs_vaps: list[float],
                              veps_list: list[np.ndarray],
                              condicions_inicials: np.ndarray,
                              t_steps: int,
                              t_span: Tuple[float, float]
                              ) -> None:
    """Dibuixa els clsuters trobats per cada radi d'esparsificació 
    que generi un màxim relatiu de les diferències entre VAPs consecutius."""
    for num, index in enumerate(indexs_max_rel):
        radi = radis[index]
        percent = sparsificacions[index]
        n_clusters = nums_clusters[index]
        diff_max = diffs_vaps[index]
        veps_index = veps_list[index]
        print(f"Maxim_relatiu_num {num}\n"
            f"Radi: {radi:.3f}, Esparsificació: {percent:.2%}, "
            f"Clusters: {n_clusters}, Max Eigen gap: {diff_max:.5e}")
        labels = troba_clusters(n_clusters, veps_index)
        grafica_clusters(condicions_inicials, labels, n_clusters, radi, percent, t_steps, t_span)