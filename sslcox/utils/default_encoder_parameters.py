import torch

def get_default_encoder_parameters(latent_features:int=10):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else device)

    return {
        'latent_features': latent_features,
        'hidden_layers': [5000],
        'layer_activation': 'Tanh',
        'use_batch_norm': True,
        'device': device, 
        'n_epochs': 10000,
        'batch_size': 256,
        'l2_penalty': 0.0001,
        'KLD_ratio': 0.0001, 
        'orthogonal_ratio': 0.001,
        'start_time_sigma': 1e-2, # Default
        'end_time_sigma': 1e-8, # Default
        'n_epochs_sigma': 1000, # Default
        'early_stopping': True, # Default
        'no_patience_before': 10, # Default
        'patience': 100, # Default
        'val_size': 0.1, # Default
        'cox_additional_evals': 5,
        'lr': 0.0001,
        'cosine_Tmax': 50, # Default
        'optimizer_name': "Adam", # Default
        'verbose': 0, # Default
        'seed': 0, # Default
    }