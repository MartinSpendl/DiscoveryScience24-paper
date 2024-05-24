"""
Running model training for clustering on METABRIC and TCGA datasets
"""

import os
import sys
sys.path.append(f"../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
import json

from sslcox.models.other_models import NoEmbedding, PCAEmbedding
from sslcox.models.VariationalAE import VAE
from sslcox.data.load_CCLE_with_GDSC2 import load_CCLE_with_GDSC2
from sslcox.metrics.morans_measure import moran_measure
from sslcox.data.load_datasets import load_METABRIC, load_TCGA_clustering
from sslcox.utils.run_ssGSEA import calculate_ssGSEA_scores
from sslcox.utils.storing_json_results import save_json_results, load_json_results
from sslcox.utils.default_encoder_parameters import get_default_encoder_parameters
from sslcox.utils.model_evaluation import signal_training_done, create_DS_folder, create_CV_folder, create_MODEL_folder, model_already_trained

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

import optuna
import pickle
import argparse
    

def main(EXPRESSIONS, HYPER_PARAMETER_OPTIMIZATION, OVERRIDE_RESULTS, LATENT_FEATURES):

    N_TRIALS = 100

    DS_DIR = f'{EXPRESSIONS}-clustering-optuna'
    CV_DIR = lambda cv: f'CV-{cv}'
    MODEL_DIR = lambda m: f'{m}-model-results'

    encoder_parameters = get_default_encoder_parameters(latent_features=LATENT_FEATURES)

    if EXPRESSIONS != 'METABRIC':
        expressions, _ = load_TCGA_clustering(EXPRESSIONS)
    else:
        expressions, _ = load_METABRIC()

    L1000 = pd.read_csv('../data/genesets/GSE92742_Broad_LINCS_gene_info_delta_landmark.txt', sep='\t', index_col=[0])['pr_gene_symbol'].values
    X = expressions[[g for g in expressions.columns if g in L1000]]

    studies = {'vae-mse': [], 'vae-cox': [], 'vae-div': []}

    create_DS_folder(DS_DIR)

    create_CV_folder(DS_DIR, CV_DIR(1))

    print(f'Training {EXPRESSIONS}')
    encoder_parameters['seed'] = 0

    X_train = X.copy()
    
    std_scaler = StandardScaler()
    X_train = pd.DataFrame(std_scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

    # Train VAE
    encoder_models = {
        'no-embedding': NoEmbedding(),
        'vae-mse': VAE('mse', **encoder_parameters),
        'vae-cox': VAE('cox', **encoder_parameters),
        'vae-div': VAE('div', **encoder_parameters),
        'pca-emb': PCAEmbedding(encoder_parameters['latent_features']),
    }
    for name, model in encoder_models.items():

        create_MODEL_folder(DS_DIR, CV_DIR(1), MODEL_DIR(name))

        study = None
        if model_already_trained(DS_DIR, CV_DIR(1), MODEL_DIR(name)) and not OVERRIDE_RESULTS:
            continue

        if name in ['vae-mse', 'vae-div', 'vae-cox'] and HYPER_PARAMETER_OPTIMIZATION:
            
            X_train_hyper, X_test_hyper = train_test_split(X_train, shuffle=True, random_state=0)
            objective_params = {"X_train_hyper": X_train_hyper, "X_test_hyper": X_test_hyper, "model_type": name.split('-')[1]}
            
            def objective(trial):
                params_ = encoder_parameters.copy()
                params_.update({
                    'lr': trial.suggest_float('lr', 0.00005, 0.001, log=True),
                    'l2_penalty': trial.suggest_float('l2_penalty', 0.001, 1000, log=True),
                    'orthogonal_ratio': trial.suggest_float('orthogonal_ratio', 0.00001, 10000, log=True),
                    'hidden_layers': trial.suggest_categorical('hidden_layers', [[5000], [500]*5, [1000, 500, 200, 100]]),
                    'cosine_Tmax': trial.suggest_float('cosine_Tmax', 10, 200),
                })
                model = VAE(objective_params['model_type'], **params_)
                model.fit(objective_params['X_train_hyper'])
                return model.evaluate(objective_params['X_test_hyper'])
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=N_TRIALS)

            print(study.best_params)
            params_ = encoder_parameters.copy()
            params_.update(study.best_params)
            studies[name].append(study)

            encoder_models[name] = VAE(name.split('-')[1], **params_)
            encoder_models[name].fit(X_train)

        else:
            model.fit(X_train)
        
        ### writing results and storing model
        PATH_TO_DIR = f'../data/training-results/{DS_DIR}/{CV_DIR(1)}/{MODEL_DIR(name)}'
        
        encoder_models[name].predict_latent(X_train).reset_index().to_csv(f'{PATH_TO_DIR}/X_train_latent.tsv', sep='\t', index=False)

        if name[:3] == 'vae':
            torch.save(encoder_models[name].model, f'{PATH_TO_DIR}/torch_model.pt')
            
            with open(f'{PATH_TO_DIR}/study.pickle', 'wb') as f:
                pickle.dump(study, f)

        with open(f'{PATH_TO_DIR}/VAE_model_object.pickle', 'wb') as f:
            pickle.dump(encoder_models[name], f)
        
        signal_training_done(PATH_TO_DIR)



if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Model training for clustering')

    # Adding arguments
    parser.add_argument('--expressions', type=str, default='METABRIC', choices=['BRCA', 'LGG', 'KIRP', 'METABRIC'],
                        help='Dataset to use: BRCA, LGG, KIRP, METABRIC')
    parser.add_argument('--hyper-parameter-optimization', action='store_true', help='Enable hyper-parameter optimization')
    parser.add_argument('--override', action='store_true', help='Override existing results')
    parser.add_argument('--latent-features', type=int, default=50, help='Enable hyper-parameter optimization')


    # Parse arguments
    args = parser.parse_args()
    
    # create data/training-results if not existent
    if 'training-results' not in os.listdir('../data'):
        os.mkdir('../data/training-results')

    # Call the main function with the parsed arguments
    for expressions in ['BRCA', 'LGG', 'KIRP', 'METABRIC']:
        main(expressions, args.hyper_parameter_optimization, args.override, args.latent_features)
