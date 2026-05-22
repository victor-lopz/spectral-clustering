# src/__init__.py

from .datatypes import (ParametresGenerals, SpectralAnalysisResult)

from .ode import (edo_duffing_autonom, edo_duffing_no_autonom)

from .trajectories import (
    generar_condicions_inicials,
    generar_trajectories,
    calcula_matriu_pesos,
)

from .spectral import (
    sparcify_with_tol,
    sparcify,
    calcula_vaps,
    calcula_num_clusters_i_max_eigengap,
    troba_clusters,
    calcula_indicadors_vs_radis,
    troba_indexs_max_rel,
    grafica_clusters_maxs_rel,
)

from .plotting import (
    grafica_trajectories,
    grafica_punts,
    grafica_eigengaps_vs_radi,
    grafica_clusters,
    grafica_eigenvalues_vs_index,
    grafica_eigengaps_vs_index,
)

from .metrica_coherent import (grafica_coherencia)

from .utils import (
    troba_radi_optim,
    aplica_spectral_clustering_optim,
)