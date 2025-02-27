from sklearn.cluster import KMeans
import numpy as np

def cluster_data(X, n_clusters: int = 3, random_state: int = 42):
    """
    Aplica K-means a los datos X.
    Devuelve los labels (asignación de cluster para cada fila).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    labels = kmeans.labels_
    return labels

def find_optimal_clusters(X, max_k: int = 10, random_state: int = 42):
    """
    (Ejemplo opcional) Usa el método 'elbow' para sugerir un número de clusters
    entre 1 y max_k. Devuelve el valor de k con la menor inercia o uno que
    optimice un criterio heurístico.
    """
    inertias = []
    K_range = range(1, max_k+1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Un método sencillo: escoger el codo en la curva 'inertias'
    # (esto requiere un análisis heurístico).
    # Para simplificar, devolvemos el k que genera el mayor decremento relativo.
    
    diffs = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
    # Se escoge aquel i con mayor diffs[i-1]
    best_k = diffs.index(max(diffs)) + 2  # +2 por el offset del range
    return best_k, inertias
