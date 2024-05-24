"""Functions for loading the data"""
import os
import json
import pandas as pd

file_location = os.path.dirname(__file__)
DATA_DIR = os.path.join(file_location, "../../data/processed")


def load_genesets():
    """
    Load hallmark genesets from the 'hallmark_genesets.json' file.

    Return
    ------
    dict
        Dictionary with keys as geneset names and value as list of genes.
    """

    with open(f"{DATA_DIR}/hallmark_genesets.json", "r") as f:
        content = json.load(f)
        genesets = {gs: values["geneSymbols"] for gs, values in content.items()}
    
    l1000_genes = pd.read_csv(f"{DATA_DIR}/../raw/L1000.tsv", sep="\t", index_col=[0], skiprows=7)
    genesets["L1000"] = [g for g in l1000_genes["pr_gene_symbol**"].values]

    return genesets
