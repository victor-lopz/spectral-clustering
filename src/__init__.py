# src/__init__.py

from .ode import (edo_duffing_autonom, edo_duffing_no_autonom)

from .trajectories import (generar_condicions_inicials, generar_trajectories, 
                           calcula_matriu_pesos)

from .spectral import (calcula_estadistics, sparcify, calcula_matriu_grau, calcula_vaps, sparcify_with_tol,
calcula_num_clusters_i_max_eigengap, calcula_indicadors_vs_radis, troba_indexs_max_rel,
troba_clusters, grafica_clusters_maxs_rel)

from .plotting import (grafica_trajectories, grafica_punts, grafica_eigengaps_vs_radi,
grafica_clusters, grafica_eigenvalues_vs_index, grafica_eigengaps_vs_index,
grafica_eigengaps_vs_radi_for_slides)

from .datatypes import (ParametresGenerals, SpectralAnalysisResult)