# src/__init__.py

from .ode import edo_duffing
from .trajectories import generar_condicions_inicials, generar_trajectories
from .spectral import (calcula_matriu_pesos, calcula_estadistics, 
imprimeix_estadistics, sparcify, calcula_matriu_grau, calcula_vaps, 
calcula_num_clusters_i_max_eigengap, calcula_diffs_vs_radis, troba_clusters)
from .plotting import (grafica_circumferencia, grafica_regio, 
grafica_regions_A_B, grafica_trajectories, grafica_punts, grafica_eigengaps_vs_radi,
grafica_clusters)