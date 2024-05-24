import os
import numpy as np
import pandas as pd

from sslcox.data.load_CCLE_with_GDSC2 import load_CCLE_with_GDSC2


def load_CCLE():

    dt, drugs = load_CCLE_with_GDSC2()
    meta = pd.read_excel('../data/raw/Cell_Lines_Details.xlsx', skipfooter=1, index_col=1)

    return dt, meta.loc[dt.index], drugs.loc[dt.index]

def load_METABRIC():

    expressions = pd.read_csv('../data/raw/brca_metabric/data_mrna_illumina_microarray.txt', sep='\t', index_col=[0]).drop('Entrez_Gene_Id', axis=1).T.fillna(0)
    expressions = expressions.loc[:, ~expressions.columns.duplicated()]
    meta = pd.read_csv('../data/raw/brca_metabric/data_clinical_patient.txt', sep='\t', index_col=[0], skiprows=4).loc[expressions.index.values]
    meta = meta[meta['CLAUDIN_SUBTYPE'].isin(['Basal', 'Her2', 'LumA', 'LumB'])]

    return expressions.loc[meta.index.values], meta


def load_TCGA_survival_data(tcga_project: str = "LGG") -> pd.DataFrame:
    """
    Loads TCGA expression and meta data for survival.
    The last two columns of the DataFrame are "time" and "event"

    Arguments
    ---------
    tcga_project: str
        The abbriviation of the TCGA project to load the data.

    Return
    ------
    pd.DataFrame
        A DataFrame with rows as samples and columns as genes. The last two
        columns are survival time ('time') ane event occurance ('event')
    """
    
    DATA_DIR = '../data/processed'

    assert f"TCGA-{tcga_project}-expressions.tsv" in os.listdir(
        DATA_DIR
    ), f"TCGA project '{tcga_project}' data is missing in the data folder."

    expressions = pd.read_csv(
        f"{DATA_DIR}/TCGA-{tcga_project}-expressions.tsv",
        sep="\t",
        index_col=[0],
    ).T
    metadata = pd.read_csv(
        f"{DATA_DIR}/TCGA-{tcga_project}-metadata.tsv", sep="\t", index_col=[0]
    )

    return pd.concat((expressions, metadata), axis=1)

def load_TCGA_clustering(tcga_project:str='BRCA'):
    """Return project expressions and cluster metadata"""

    target_columns = {
        'BRCA': 'breast_carcinoma_estrogen_receptor_status',
        'LGG': 'ldh1_mutation_found',
        'KIRP': 'tumor_type',
    }

    expressions = pd.read_csv(f'../data/raw/TCGA-clustering/HiSeq-{tcga_project}', sep='\t', index_col=[0]).T
    meta = pd.read_csv(f'../data/raw/TCGA-clustering/TCGA.{tcga_project}.sampleMap-{tcga_project}_clinicalMatrix', sep='\t', index_col=[0])

    meta = meta[target_columns[tcga_project]].loc[[i for i in meta.index.values if i in expressions.index.values]]

    return expressions, meta
