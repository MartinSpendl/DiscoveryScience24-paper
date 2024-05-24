import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def moran_measure(tsne_data:np.ndarray, cluster_labels:np.ndarray, percent_distance=0.05):
    """
    Moran's measure or I.
    
    Arguments:
    ----------
    tsne_data: np.ndarray
        2D array of tsne coordinates
    cluster_labels: np.ndarray
        1D array of binary labels if in or not in the cluster
    
    Returns:
    --------
    float between -1 and 1 (0 being random and 1 being perfect clustering)
    """

    distances = np.sqrt(np.sum(np.square(tsne_data.reshape(tsne_data.shape[0], 1, 2) - tsne_data.reshape(1, tsne_data.shape[0], 2)), axis=-1))
    span_std = percent_distance * (distances.max() - distances.min())
    distances /= span_std
    weights = norm().pdf(distances) / norm.pdf(0)
    weights[np.arange(len(weights)), np.arange(len(weights))] = 0

    diff_mean = cluster_labels - np.mean(cluster_labels)

    I = len(cluster_labels) * np.sum(weights * (diff_mean.reshape(-1,1) * diff_mean.reshape(1,-1))) / (np.sum(weights) * np.sum(diff_mean**2))

    return I

def bootstrap_moran_I(tsne_data, meta_data, M=100):

    measures = []
    unique_values = list(np.unique(meta_data))

    np.random.seed(0)
    for i in tqdm(range(M)):
        ids = np.random.choice(np.arange(len(tsne_data)), size=len(tsne_data))

        measures += [ np.mean([
            moran_measure(tsne_data[ids], np.array([k == unique_values.index(i) for i in meta_data[ids]]).astype(int), percent_distance=0.05)
            for k in range(len(unique_values))
        ])]

    mu = np.mean([
            moran_measure(tsne_data, np.array([k == unique_values.index(i) for i in meta_data]).astype(int), percent_distance=0.05)
            for k in range(len(unique_values))
        ])
    
    return mu, np.std(measures), np.std(measures)/np.sqrt(M)