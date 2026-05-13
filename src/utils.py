import numpy as np
import scipy.linalg
import scipy.spatial.distance
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def troba_radi_optim(trajectories: np.ndarray):
    """
    Realitza un escombrat (sweep) del radi r per trobar el valor que
    maximitza l'eigengap normalitzat i calcula el percentatge d'esparsificació.
    """
    num_trajectories, t_steps, dimensio = trajectories.shape
    
    # 1. Calcular la matriu de distàncies mitjanes
    distancies_1d = 0.5 * scipy.spatial.distance.pdist(trajectories[:, 0, :])
    for t in range(1, t_steps - 1):
        distancies_1d += scipy.spatial.distance.pdist(trajectories[:, t, :])
    distancies_1d += 0.5 * scipy.spatial.distance.pdist(trajectories[:, -1, :])
    
    distancies_1d = distancies_1d / (t_steps - 1)
    matriu_distancies = scipy.spatial.distance.squareform(distancies_1d)
    
    # Nombre total d'arestes possibles (excloent la diagonal / nodes amb ells mateixos)
    arestes_totals = num_trajectories * (num_trajectories - 1)
    
    # 2. Definir l'escombrat de radis (r de 0.25 a 3.0)
    r_valors = np.linspace(0.25, 3.0, 10)
    gaps_normalitzats = []
    
    millor_r = None
    millor_gap = -1
    millor_k = 0
    millor_sparsification = 0.0
    
    print("Iniciant l'escombrat de paràmetres...")
    
    for r in r_valors:
        # 3. Construir la matriu de pesos W sparsificada
        W = np.zeros_like(matriu_distancies)
        mask = (matriu_distancies > 0) & (matriu_distancies < r)
        W[mask] = 1.0 / matriu_distancies[mask]
        
        # Càlcul del percentatge d'esparsificació (% d'arestes ELIMINADES)
        arestes_mantingudes = np.sum(mask)
        sparsification_pct = 100.0 * (1.0 - (arestes_mantingudes / arestes_totals))
        
        # 4. Aplicar l'offset a la diagonal
        w_max = np.max(W) if np.max(W) > 0 else 1.0
        np.fill_diagonal(W, w_max * 10**7) 
        
        # 5. Calcular el Laplacià Normalitzat
        graus = np.sum(W, axis=1)
        graus[graus == 0] = 1e-10 # Evitar divisió per zero
        d_inv_sqrt = 1.0 / np.sqrt(graus)
        
        W_norm = W * np.outer(d_inv_sqrt, d_inv_sqrt)
        L_N = np.eye(num_trajectories) - W_norm
        
        # 6. Calcular valors propis
        eigenvalues = scipy.linalg.eigvalsh(L_N)
        
        # 7. Calcular l'eigengap normalitzat
        gaps = np.diff(eigenvalues) 
        max_gap = np.max(gaps)
        k_gap = np.argmax(gaps) + 1 
        
        rang_lambda = eigenvalues[-1] - eigenvalues[0]
        gap_normalitzat = max_gap / rang_lambda if rang_lambda > 0 else 0
        
        gaps_normalitzats.append(gap_normalitzat)
        
        # Guardar el millor resultat
        if gap_normalitzat > millor_gap:
            millor_gap = gap_normalitzat
            millor_r = r
            millor_k = int(k_gap)
            millor_sparsification = sparsification_pct

    print(f"Radi òptim trobat: r={millor_r:.3f} | k={millor_k} clústers | W is {millor_sparsification:.2f}% sparsified")
    # (Opcional) Dibuixar el gràfic de l'escombrat
    plt.figure(figsize=(10, 6))
    plt.plot(r_valors, gaps_normalitzats, marker='D', linestyle='--', color='purple', alpha=0.7)
    assert millor_r is not None, "No s'ha trobat un radi òptim vàlid."
    plt.axvline(millor_r, color='gray', linestyle=':', label=f'Pic màxim a r={millor_r:.3f}')
    plt.xlabel('Radi r (Sparsification radius)', fontsize=12)
    plt.ylabel('Gap ratio |MaxGap| / (λ_max - λ_min)', fontsize=12)
    plt.title('Gap ratio versus r for the asymmetric Duffing oscillator', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return millor_r, millor_k, millor_sparsification


def aplica_spectral_clustering_optim(trajectories: np.ndarray, optimal_r: float, optimal_k: int) -> np.ndarray:
    """
    Aplica l'algorisme de Spectral Clustering definitiu utilitzant el 
    radi òptim i el nombre de clústers òptims trobats en l'escombrat.
    """
    num_trajectories, t_steps, dimensio = trajectories.shape
    
    # 1. Calcular la matriu de distàncies mitjanes
    distancies_1d = 0.5 * scipy.spatial.distance.pdist(trajectories[:, 0, :])
    for t in range(1, t_steps - 1):
        distancies_1d += scipy.spatial.distance.pdist(trajectories[:, t, :])
    distancies_1d += 0.5 * scipy.spatial.distance.pdist(trajectories[:, -1, :])
    
    distancies_1d = distancies_1d / (t_steps - 1)
    matriu_distancies = scipy.spatial.distance.squareform(distancies_1d)
    
    # 2. Construir la matriu de pesos W sparsificada amb el radi òptim
    W = np.zeros_like(matriu_distancies)
    mask = (matriu_distancies > 0) & (matriu_distancies < optimal_r)
    W[mask] = 1.0 / matriu_distancies[mask]
    
    # 3. Aplicar l'offset a la diagonal (per estabilitat numèrica)
    w_max = np.max(W) if np.max(W) > 0 else 1.0
    np.fill_diagonal(W, w_max * 10**7)
    
    # 4. Calcular el Laplacià Normalitzat
    graus = np.sum(W, axis=1)
    # Evitar divisions per zero en cas de nodes aïllats restants
    graus[graus == 0] = 1e-10 
    d_inv_sqrt = 1.0 / np.sqrt(graus)
    
    # W_norm = D^(-1/2) * W * D^(-1/2)
    W_norm = W * np.outer(d_inv_sqrt, d_inv_sqrt)
    L_N = np.eye(num_trajectories) - W_norm
    
    # 5. Calcular vectors propis (usant eigh per a matrius simètriques)
    eigenvalues, eigenvectors = scipy.linalg.eigh(L_N)
    
    # # Ens assegurem que estan ordenats de menor a major
    # idx = np.argsort(eigenvalues)
    # eigenvectors = eigenvectors[:, idx]
    
    # 6. Extreure els primers 'optimal_k' vectors propis per formar la matriu U
    # Cada fila d'aquesta matriu representa una trajectòria a l'espai espectral
    U = eigenvectors[:, :optimal_k]
    
    # 7. Aplicar K-Means a l'espai espectral
    print(f"Aplicant K-Means demanant exactament {optimal_k} clústers...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(U)
    
    return labels