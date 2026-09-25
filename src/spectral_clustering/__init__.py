# src/spectral_clustering/__init__.py

from .datatypes import SpectralAnalysisResult, SpectralClusteringConfig
from .ode import ode_autonomous_duffing, ode_non_autonomous_duffing
from .plotting import (
    plot_clusters,
    plot_coordinates,
    plot_eigengaps_vs_index,
    plot_eigengaps_vs_sparsity,
    plot_eigenvalues_vs_index,
    plot_trajectory_paths,
)
from .spectral import (
    calculate_degree_matrix,
    calculate_eigenvalues,
    calculate_num_clusters_and_max_eigengap,
    calculate_spectral_indicators,
    calculate_weight_statistics,
    find_clusters,
    find_local_maxima,
    plot_clusters_local_maxs,
    sparsify,
    sparsify_with_radius,
)
from .trajectories import (
    compute_weight_matrix,
    generate_initial_conditions,
    simulate_trajectories,
)

__all__ = [
    "SpectralAnalysisResult",
    "SpectralClusteringConfig",
    "calculate_degree_matrix",
    "calculate_eigenvalues",
    "calculate_num_clusters_and_max_eigengap",
    "calculate_spectral_indicators",
    "calculate_weight_statistics",
    "compute_weight_matrix",
    "find_clusters",
    "find_local_maxima",
    "generate_initial_conditions",
    "ode_autonomous_duffing",
    "ode_non_autonomous_duffing",
    "plot_clusters",
    "plot_clusters_local_maxs",
    "plot_coordinates",
    "plot_eigengaps_vs_index",
    "plot_eigengaps_vs_sparsity",
    "plot_eigenvalues_vs_index",
    "plot_trajectory_paths",
    "simulate_trajectories",
    "sparsify",
    "sparsify_with_radius",
]
