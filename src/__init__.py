# src/__init__.py

from .datatypes import ParametresGenerals, SpectralAnalysisResult
from .ode import edo_duffing_autonom, edo_duffing_no_autonom
from .plotting import (
    plot_clusters,
    plot_eigengaps_vs_index,
    grafica_eigengaps_vs_radi,
    plot_eigenvalues_vs_index,
    plot_coordinates,
    plot_trajectory_paths,
)
from .spectral import (
    calcula_estadistics,
    calcula_indicadors_vs_radis,
    calcula_matriu_grau,
    calcula_num_clusters_i_max_eigengap,
    calcula_vaps,
    grafica_clusters_maxs_rel,
    sparsify,
    sparsify_with_tol,
    troba_clusters,
    troba_indexs_max_rel,
)
from .trajectories import (
    calcula_matriu_pesos,
    generar_condicions_inicials,
    generar_trajectories,
)

__all__ = [
    "ParametresGenerals",
    "SpectralAnalysisResult",
    "calcula_estadistics",
    "calcula_indicadors_vs_radis",
    "calcula_matriu_grau",
    "calcula_matriu_pesos",
    "calcula_num_clusters_i_max_eigengap",
    "calcula_vaps",
    "edo_duffing_autonom",
    "edo_duffing_no_autonom",
    "generar_condicions_inicials",
    "generar_trajectories",
    "plot_clusters",
    "grafica_clusters_maxs_rel",
    "plot_eigengaps_vs_index",
    "grafica_eigengaps_vs_radi",
    "plot_eigenvalues_vs_index",
    "plot_coordinates",
    "plot_trajectory_paths",
    "sparsify",
    "sparsify_with_tol",
    "troba_clusters",
    "troba_indexs_max_rel",
]
