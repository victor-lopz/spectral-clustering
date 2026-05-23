# %% [markdown]
# ### Algorisme d'agrupament espectral aplicat al sistema de Duffing autònom

# %%
import sys
sys.path.append("..")
from src import *
import numpy as np
np.set_printoptions(precision=3, suppress=True)
%load_ext autoreload
%autoreload 2

# %%
params = ParametresGenerals(
    dimensio = 2,
    t_span = (0, 4 * np.pi),
    t_steps = 300,
    x_min = -1.6,
    x_max = 1.6,
    y_min = -1.0,
    y_max = 1.0,
    espai_entre_punts = 0.04,
    max_clusters = 50,
    num_radis = 50
)

# %%
condicions_inicials = generar_condicions_inicials(params)
trajectories = generar_trajectories(edo_duffing_autonom, condicions_inicials, params)
print("(Num trajectories, t_steps, dimensio) =", trajectories.shape)

# %%
matriu_pesos = calcula_matriu_pesos(trajectories)

# %%
pes_max = np.max(matriu_pesos)
print(f"Pes màxim a la matriu de pesos: {pes_max:.3f}")
constant_diagonal = pes_max * 1e7

# %% [markdown]
# ##### Opció A: esparsificar al 90%

# %%
matriu_similaritat_W, radi_esparsificacio, percentatge_esparsificacio = \
      sparcify(matriu_pesos, percent=90)
print(f"S'ha obtingut una esparsificació del {percentatge_esparsificacio*100:.0f}% "
      f"usant {radi_esparsificacio:.3f} com a radi d'esparsificació.")
np.fill_diagonal(matriu_similaritat_W, constant_diagonal)
print("matriu_similaritat_W =\n", matriu_similaritat_W)
vaps, veps = calcula_vaps(matriu_similaritat_W, params.max_clusters)
print("veps.shape =", veps.shape)
print(len(vaps),"vaps =",
      np.array2string(vaps,formatter={'float_kind':lambda x:f'{x:.1e}'}))
grafica_eigenvalues_vs_index(vaps, subfolder="autonom")
grafica_eigengaps_vs_index(vaps, subfolder="autonom")
num_clusters, diff_max = calcula_num_clusters_i_max_eigengap(vaps)
labels = troba_clusters(num_clusters, veps)
grafica_clusters(condicions_inicials, labels, num_clusters, radi_esparsificacio, 
                 percentatge_esparsificacio, params, subfolder="autonom")

# %% [markdown]
# ##### Opció B: mètode adaptatiu. Es trien els radis d'esparsificació que maximitzen l'eigengap màxim

# %%
result = calcula_indicadors_vs_radis(matriu_pesos, constant_diagonal, params)

# %%
indexs_max_rel = troba_indexs_max_rel(result.normalized_eigengaps)
print("indexs_max_rel =", indexs_max_rel)
print("radis[indexs_max_rel] =", result.radis[indexs_max_rel])
grafica_clusters_maxs_rel(indexs_max_rel, result, condicions_inicials, 
                          params, subfolder="autonom_maxs_rels")

# %%
grafica_eigengaps_vs_radi(result, params, indexs_max_rel=indexs_max_rel, 
                          subfolder="autonom_maxs_rels")


