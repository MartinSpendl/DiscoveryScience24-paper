import pandas as pd
from sklearn.decomposition import PCA

class PCAEmbedding:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components)
    
    def fit(self, X):
        self.pca.fit(X)
    
    def predict_latent(self, X):
        return pd.DataFrame(self.pca.transform(X), index=X.index)

class NoEmbedding:
    def __init__(self, **kwargs):
        pass
    def fit(self, X):
        pass
    def predict_latent(self, X):
        return X.copy()
