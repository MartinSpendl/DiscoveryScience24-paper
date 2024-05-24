"""Load CCLE cell line data with GDSC2 drug screening data"""

import numpy as np
import pandas as pd
from collections import namedtuple

def load_drug_data():

    raw = pd.read_csv('../data/raw/GDSC2_fitted_dose_response.csv')

    temp = pd.DataFrame(index=np.unique(raw['COSMIC_ID'].values), columns=np.unique(raw['DRUG_NAME'].values))

    for i in range(len(raw)):
        row = raw.iloc[i]
        temp.loc[row['COSMIC_ID'], row['DRUG_NAME']] = row['LN_IC50'] 
    
    return temp

def load_CCLE_expression_data():

    data = pd.read_csv('../data/raw/Cell_line_RMA_proc_basalExp.txt', sep='\t')
    data = data.dropna(axis=0).set_index('GENE_SYMBOLS').drop(labels=['GENE_title'], axis=1).T
    
    duplicated = [i[:-2] for i in data.index.values if i[-2:] == '.1']
    data = data.loc[[i for i in data.index.values if i not in duplicated]]
    data.index = [int(i.split('.')[1]) for i in data.index.values if i not in duplicated]
    
    return data

def load_CCLE_with_GDSC2():

    drug_data = load_drug_data()

    expressions = load_CCLE_expression_data()

    joined_index = joined_index = expressions.index.intersection(drug_data.index)

    CCLE = namedtuple('CCLE', ['expressions', 'IC50'])

    return CCLE(
        expressions.loc[joined_index],
        drug_data.loc[joined_index]
    )