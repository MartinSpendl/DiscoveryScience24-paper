import pickle
import os
import numpy as np
import pandas as pd
from sslcox.models.VariationalAE import VAE

file_location = os.path.dirname(__file__)
DATA_DIR = os.path.join(file_location, "../../data/model-checkpoints")

def save_VAE_model(vae_object, filename):

    while filename in os.listdir(DATA_DIR):
        filename += '-1'

    with open(f'{DATA_DIR}/{filename}', 'wb') as f:
        pickle.dump(vae_object, f)

def load_VAE_model(filename):

    assert filename in os.listdir(DATA_DIR), f"{filename} does not exist"

    with open(f'{DATA_DIR}/{filename}', 'rb') as f:
        vae_object = pickle.load(f)
    
    return vae_object
