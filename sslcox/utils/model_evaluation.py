import os
import numpy as np
import pandas as pd
from sslcox.data.genesets import load_genesets
from sslcox.data.load_CCLE_with_GDSC2 import load_CCLE_with_GDSC2
from sslcox.models.other_models import NoEmbedding, PCAEmbedding
from sslcox.models.VariationalAE import VAE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_experiment_filename(ep):
    """
    Arguments:
    ep: dict
        Experiment parameters as ep
    
    Returns:
    str
        Filename of the experiment without location.
    """

    filename = (
        f"experiment-ds={ep['dataset']}-tts={ep['train_test_seed']}-gs={ep['gene_selection']}-norm={ep['normalization_method']}-" + 
            f"enc={ep['encoder_method']}-encseed={ep['encoder_parameters']['seed']}-lat={ep['latent_features']}-down={ep['downstream_method']}.tsv"
    )

    return filename

def experiment_already_perfomed(experiment_parameters, path_to_store):
    
    filename = get_experiment_filename(experiment_parameters)

    return filename in os.listdir(path_to_store)

def store_results(results, experiment_parameters, path_to_store):

    filename = get_experiment_filename(experiment_parameters)

    results.reset_index().to_csv(f"{path_to_store}/{filename}", sep='\t', index=False)

def get_dataset(dataset):

    if dataset == "CCLE":
        return load_CCLE_with_GDSC2()
    
    else:
        raise KeyError(f'Unknown dataset type: {dataset}.')

def get_train_test_split(data, train_test_seed):

    train_ids, test_ids = train_test_split(np.arange(len(data.expressions)), test_size=0.2, random_state=train_test_seed)

    X = data.expressions
    y = data[1] # IC50 or else

    return (
        X.iloc[train_ids],
        X.iloc[test_ids],
        y.iloc[train_ids],
        y.iloc[test_ids]
    )

def get_reduced_datasets(train_X_data, test_X_data, gene_selection):

    if gene_selection == 'L1000':
        return select_L1000_genes(train_X_data, test_X_data)
    
    elif 'high-variance' in gene_selection:
        return select_high_variance_genes(train_X_data, test_X_data, gene_selection)
    
    else:
        raise KeyError(f'Unknown gene selection method: {gene_selection}.')
    
def select_L1000_genes(train_X_data, test_X_data):

    L1000 = load_genesets()['L1000']
    L1000 = [g for g in L1000 if g in train_X_data.columns]

    return train_X_data[L1000], test_X_data[L1000]

def select_high_variance_genes(train_X_data, test_X_data, gene_selection):

    n_genes = int(gene_selection.split('-')[-1])
    MAD = (train_X_data.expressions - train_X_data.expressions.mean(axis=0)).abs().mean().sort_values(ascending=False).head(n_genes)

    return train_X_data[MAD.index.values], test_X_data[MAD.index.values]

def get_normalized_data(normalization_method, train_X_reduced, test_X_reduced):

    if normalization_method == 'std':
        std_scaler = StandardScaler()
        
        X_train = pd.DataFrame(std_scaler.fit_transform(train_X_reduced), index=train_X_reduced.index, columns=train_X_reduced.columns)
        X_test = pd.DataFrame(std_scaler.transform(test_X_reduced), index=test_X_reduced.index, columns=test_X_reduced.columns)

        return X_train, X_test
    
    elif normalization_method == 'minmax':
        minmax_scaler = MinMaxScaler()
        
        X_train = pd.DataFrame(minmax_scaler.fit_transform(train_X_reduced), index=train_X_reduced.index, columns=train_X_reduced.columns)
        X_test = pd.DataFrame(minmax_scaler.transform(test_X_reduced), index=test_X_reduced.index, columns=test_X_reduced.columns)

        return X_train, X_test
    
    else:
        raise KeyError(f'Unknown normalization method: {normalization_method}.')
    
def train_encoder(encoder_method, latent_features, encoder_parameters, train_X_data):

    if encoder_method == 'NoEmbedding':
        return train_NoEmbedding_encoder()
    
    elif encoder_method == 'PCA':
        return train_PCA_encoder(latent_features, train_X_data)
    
    elif encoder_method in ['mse', 'cox', 'elbo']:
        return train_VAE_encoder(encoder_method, encoder_parameters, train_X_data)
    
    else:
        raise KeyError(f'Unknown enocder method: {encoder_method}.')

def train_NoEmbedding_encoder():
    return NoEmbedding()

def train_PCA_encoder(latent_features, train_X_data):

    pca = PCAEmbedding(latent_features)
    pca.fit(train_X_data)

    return pca

def train_VAE_encoder(encoder_method, encoder_parameters, train_X_data):

    vae = VAE(model_type=encoder_method,**encoder_parameters)
    vae.fit(train_X_data)

    return vae

def transform_to_latent_features(encoder, train_X_norm, test_X_norm):

    train_X_latent = encoder.predict_latent(train_X_norm)
    test_X_latent = encoder.predict_latent(test_X_norm)

    return train_X_latent, test_X_latent

def train_downstream_model(downstream_method, downstream_parameters, train_X_latent, train_y_data):
    
    if downstream_method == 'KNN':
        return train_KNN_model(downstream_parameters, train_X_latent, train_y_data)
    
    elif downstream_method == 'Ridge':
        return train_Ridge_model(downstream_parameters, train_X_latent, train_y_data)
    
    else:
        raise KeyError(f'Unknown downstream method: {downstream_method}.')

def get_druglist(train_y_data):
    return (train_y_data.fillna(0) > 0).sum(axis=0).sort_values(ascending=False).index.values

def train_KNN_model(downstream_parameters, train_X_latent, train_y_data):

    drug_list = get_druglist(train_y_data)

    models = {}

    for drug in drug_list[:downstream_parameters['number_drugs']]:
        
        y = train_y_data[drug].dropna()
        X = train_X_latent.loc[y.index]
    
        KNN = KNeighborsRegressor(n_neighbors=downstream_parameters['nearest_neighbours'])
        KNN.fit(X, y)

        models[drug] = KNN
    
    return models

def train_Ridge_model(downstream_parameters, train_X_latent, train_y_data):

    drug_list = get_druglist(train_y_data)

    models = {}

    for drug in drug_list[:downstream_parameters['number_drugs']]:
        
        y = train_y_data[drug].dropna()
        X = train_X_latent.loc[y.index]
    
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10])
        ridge.fit(X, y)

        models[drug] = ridge
    
    return models

def evaluate_final_models(final_models, test_X_latent, test_y_data):

    results = {}

    for drug, model in final_models.items():
        results[f'{drug}-prediction'] = model.predict(test_X_latent)
    results = pd.DataFrame(results, index=test_X_latent.index)

    return pd.concat((results, test_y_data[list(final_models.keys())]), axis=1)

def signal_training_done(path_to_dir:str):

    with open(f'{path_to_dir}/.done', 'w') as f:
        f.write('')

def create_DS_folder(ds_dir):

    path = f'../data/training-results'

    if ds_dir in os.listdir(path):
        return
    else:
        os.mkdir(f'{path}/{ds_dir}')

def create_CV_folder(ds_dir, cv_dir):

    path = f'../data/training-results/{ds_dir}'

    if cv_dir in os.listdir(path):
        return
    else:
        os.mkdir(f'{path}/{cv_dir}')

def create_MODEL_folder(ds_dir, cv_dir, model_dir):

    path = f'../data/training-results/{ds_dir}/{cv_dir}'

    if model_dir in os.listdir(path):
        return
    else:
        os.mkdir(f'{path}/{model_dir}')

def model_already_trained(ds_dir, cv_dir, model_dir):

    path = f'../data/training-results/{ds_dir}/{cv_dir}/{model_dir}'

    return '.done' in os.listdir(path)
