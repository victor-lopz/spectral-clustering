# src/__init__.py

from .datatypes import SpectralClusteringConfig, SpectralAnalysisResult
from .ode import edo_duffing_autonom, edo_duffing_no_autonom
from .plotting import (
    plot_clusters,
    plot_eigengaps_vs_index,
    plot_eigengaps_vs_sparsity,
    plot_eigenvalues_vs_index,
    plot_coordinates,
    plot_trajectory_paths,
)
from .spectral import (
    calculate_weight_statistics,
    calculate_spectral_indicators,
    calculate_degree_matrix,
    calculate_num_clusters_and_max_eigengap,
    calculate_eigenvalues,
    plot_clusters_local_maxs,
    sparsify,
    sparsify_with_radius,
    find_clusters,
    find_local_maxima,
)
from .trajectories import (
    calcula_matriu_pesos,
    generate_initial_conditions,
    generar_trajectories,
)

__all__ = [
    "SpectralClusteringConfig",
    "SpectralAnalysisResult",
    "calculate_weight_statistics",
    "calculate_spectral_indicators",
    "calculate_degree_matrix",
    "calcula_matriu_pesos",
    "calculate_num_clusters_and_max_eigengap",
    "calculate_eigenvalues",
    "edo_duffing_autonom",
    "edo_duffing_no_autonom",
    "generate_initial_conditions",
    "generar_trajectories",
    "plot_clusters",
    "plot_clusters_local_maxs",
    "plot_eigengaps_vs_index",
    "plot_eigengaps_vs_sparsity",
    "plot_eigenvalues_vs_index",
    "plot_coordinates",
    "plot_trajectory_paths",
    "sparsify",
    "sparsify_with_radius",
    "find_clusters",
    "find_local_maxima",
]
