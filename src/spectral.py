import numpy as np
import scipy.linalg
from sklearn.cluster import KMeans

from src.datatypes import SpectralClusteringConfig, SpectralAnalysisResult
from src.plotting import plot_clusters


def calculate_weight_statistics(similarity_matrix: np.ndarray) -> dict[str, float]:
    """
    Returns a dictionary with statistics of the weights in the similarity matrix.
    The statistics include:
    - min_weight: Minimum weight in the matrix
    - median_weight: Median weight in the matrix
    - avg_weight: Average weight in the matrix
    - percentile_90: 90th percentile of the weights
    - percentile_95: 95th percentile of the weights
    - max_weight: Maximum weight in the matrix
    """
    triangular_upper = similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)]
    weight_percentiles = np.percentile(triangular_upper, [0, 50, 90, 95, 100])
    min_weight, median_weight, p90, p95, max_weight = weight_percentiles
    weight_statistics = {
        "min_weight": min_weight,
        "median_weight": median_weight,
        "avg_weight": np.mean(triangular_upper),
        "percentile_90": p90,
        "percentile_95": p95,
        "max_weight": max_weight,
    }
    return weight_statistics


def sparsify_with_radius(
    weight_matrix: np.ndarray, sparsification_radius: float
) -> tuple[np.ndarray, float]:
    """
    Returns a matrix where elements smaller than the sparsification radius
    are set to zero. Sparsification radius = tolerance.
    It also returns the percentage of sparsification achieved.
    Requirement: the diagonal of the matrix must be zero
    because it represents the similarity between a trajectory and itself.
    """
    if np.diag(weight_matrix).any():
        raise ValueError("The diagonal of the matrix must be zero.")

    is_below_tolerance = weight_matrix < sparsification_radius
    sparsified_matrix = np.where(is_below_tolerance, 0, weight_matrix)

    # Exclude the diagonal by subtracting the length of the matrix
    count_zeros = np.sum(is_below_tolerance) - len(weight_matrix)
    total_elements = weight_matrix.size - len(weight_matrix)
    sparsification_percentage = float(count_zeros / total_elements)
    return sparsified_matrix, sparsification_percentage


def get_sparsification_radius(
    weight_matrix: np.ndarray, sparsification_percent: float
) -> float:
    """
    The sparsification radius (or tolerance) is the {percent} percentile
    of the non-zero values of the {weight_matrix}.
    Since the matrix is symmetric and the diagonal is all zeros,
    we only need the upper triangular part of the matrix without the diagonal
    to compute the sparsification radius.
    """
    triangular_upper = weight_matrix[np.triu_indices(len(weight_matrix), k=1)]
    sparsification_radius = np.percentile(triangular_upper, sparsification_percent)
    return float(sparsification_radius)


def sparsify(
    weight_matrix: np.ndarray, sparsification_percent: float
) -> tuple[np.ndarray, float, float]:
    """
    Returns a sparse matrix where the chosen percentage of the smallest
    elements are set to zero.
    Requirement: the matrix must be symmetric with all zeros in the diagonal.
    """
    radius = get_sparsification_radius(weight_matrix, sparsification_percent)
    sparsified_matrix, sparsification_percent = sparsify_with_radius(
        weight_matrix, radius
    )
    return sparsified_matrix, radius, sparsification_percent


def calcula_matriu_grau(matriu_similaritat: np.ndarray) -> np.ndarray:
    """Calcula la suma de cada fila i les col·loca en una matriu diagonal."""
    return np.diag(matriu_similaritat.sum(axis=1))


def calcula_vaps(
    matriu_similaritat_W: np.ndarray, max_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna els n VAPs més petits ordenats ascendentment i
    els VEPs del problema generalitzat Lu = lambda Du.
    Requisit: les matrius L i D han de ser simètriques."""
    n = matriu_similaritat_W.shape[0]
    if n == 0:
        raise ValueError("La matriu de similaritat no pot ser buida.")
    if max_clusters <= 0:
        raise ValueError(f"cal max_clusters > 0, rebut: {max_clusters}.")
    max_index = min(max_clusters - 1, n - 1)
    matriu_grau_D = calcula_matriu_grau(matriu_similaritat_W)
    matriu_laplacia_L = matriu_grau_D - matriu_similaritat_W
    vaps, veps = scipy.linalg.eigh(
        matriu_laplacia_L, matriu_grau_D, subset_by_index=[0, max_index]
    )
    return vaps, veps


def calcula_num_clusters_i_max_eigengap(vaps: np.ndarray) -> tuple[int, float]:
    """Retorna el nombre de clusters segons l'heurística del salt espectral.
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


def calcula_indicadors_vs_radis(
    matriu_pesos: np.ndarray, constant_diagonal: float, params: SpectralClusteringConfig
) -> SpectralAnalysisResult:
    """
    Retorna una classe SpectralAnalysisResult que,
    per a cada radi d'esparsificació, conté:
    - la diferència màxima entre VAPs consecutius
    - la diferència màxima normalitzada entre VAPs consecutius
    - el nombre de clusters trobat, sempre dins del rang [1, max_clusters]
    - el percentatge d'esparsificació
    - els estadístics de la matriu de pesos, que són:
        el pes mínim, màxim, mediana, mitjà, percentils 90 i 95
    - tots els VEPs associats a cada radi d'esparsificació
    """
    estadistics = calculate_weight_statistics(matriu_pesos)
    radis = np.linspace(
        estadistics["min_weight"], estadistics["percentile_95"], params.num_radii
    )
    result = SpectralAnalysisResult(
        sparsification_radii=radis, weight_statistics=estadistics
    )
    for radi in radis:
        matriu_similaritat_W, percent = sparsify_with_radius(matriu_pesos, radi)
        result.sparsification_percents.append(percent)
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
    """Retorna els índexs dels màxims relatius del vector 'diffs'.
    S'utilitza per trobar els màxims relatius de les diferències
    entre VAPs consecutius en funció del radi d'esparsificació."""
    maxs_rels = []
    i = 1
    while i < len(diffs) - 1:
        if diffs[i - 1] < diffs[i] > diffs[i + 1]:
            maxs_rels.append(i)
            i += 2
        else:
            i += 1
    return maxs_rels


def grafica_clusters_maxs_rel(
    indexs_max_rel: list[int],
    result: SpectralAnalysisResult,
    condicions_inicials: np.ndarray,
    params: SpectralClusteringConfig,
    subfolder: str | None = None,
) -> None:
    """Dibuixa els clústers trobats per cada radi d'esparsificació que
    generi un màxim relatiu de les diferències entre VAPs consecutius."""
    for num, index in enumerate(indexs_max_rel, start=1):
        radi = result.sparsification_radii[index]
        percent = result.sparsification_percents[index]
        n_clusters = result.nums_clusters[index]
        diff_max = result.eigengaps[index]
        veps = result.veps_list[index]
        print(
            f"Màxim_relatiu_num {num}\n"
            f"Radi: {radi:.3f}, Esparsificació: {percent:.2%}, "
            f"Clústers: {n_clusters}, Max eigengap: {diff_max:.5e}"
        )
        labels = troba_clusters(n_clusters, veps)
        plot_clusters(
            condicions_inicials,
            labels,
            n_clusters,
            radi,
            percent,
            params,
            subfolder,
            filename_prefix=f"max_rel-{num}_",
        )
