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


def calculate_degree_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Returns a diagonal matrix where each diagonal element is the sum of the
    corresponding row of the similarity matrix. The non-diagonal elements are zero.

    This is the degree matrix D in spectral clustering and represents how strongly
    each trajectory is connected to the others.
    High degree means high similarity with many other trajectories,
    while low degree means that the trajectory is quite isolated.
    """
    return np.diag(similarity_matrix.sum(axis=1))


def calculate_eigenvalues(
    similarity_matrix: np.ndarray, max_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the first {max_clusters} smallest eigenvalues sorted in ascending order
    and their corresponding eigenvectors of the generalized eigenvalue problem
    Lu = lambda Du, where D is the degree matrix and L is the Laplacian matrix, defined
    as L = D - W, where W is the similarity matrix.

    Requirement: the matrices L and D must be symmetric.
    """
    n = similarity_matrix.shape[0]
    if n == 0:
        raise ValueError("The similarity matrix cannot be empty.")
    if max_clusters <= 0:
        raise ValueError(f"max_clusters > 0 is needed, but got: {max_clusters}.")
    max_index = min(max_clusters - 1, n - 1)
    degree_matrix = calculate_degree_matrix(similarity_matrix)
    laplacian_matrix = degree_matrix - similarity_matrix
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        laplacian_matrix, degree_matrix, subset_by_index=[0, max_index]
    )
    return eigenvalues, eigenvectors


def calculate_num_clusters_and_max_eigengap(vaps: np.ndarray) -> tuple[int, float]:
    """
    Returns the number of clusters according to the spectral gap heuristic.
    This rule states that the number of clusters is the index k where the difference
    between vaps[k] and vaps[k-1] is greatest. In other words, it is the argument of the
    maximum of consecutive differences of ordered eigenvalues.

    It also returns the value of the greatest difference (maximum eigengap).

    We add 1 because of zero-based indexing.
    We add 1 again to include the cluster of incoherent states.
    """
    diffs = np.diff(vaps)
    k = int(np.argmax(diffs))
    num_clusters = k + 2
    max_eigengap = diffs[k]
    return num_clusters, max_eigengap


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
        vaps, veps = calculate_eigenvalues(matriu_similaritat_W, params.max_clusters)
        num_clusters, max_eigengap = calculate_num_clusters_and_max_eigengap(vaps)
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
