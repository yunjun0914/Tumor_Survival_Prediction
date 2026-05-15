import glob
import os

import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'Data', 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'Data', 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')


def load_expression(filepath):
    """Load expression TSV (genes x samples) and transpose to samples x genes."""
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    return df.T


def load_survival(filepath):
    """Load survival TSV with case_id as index."""
    return pd.read_csv(filepath, sep='\t', index_col='case_id')


def find_file(directory, *keywords):
    """Find a .tsv file whose name contains all keywords (case-insensitive)."""
    for path in glob.glob(os.path.join(directory, '*.tsv')):
        name = os.path.basename(path).lower()
        if all(kw.lower() in name for kw in keywords):
            return path
    raise FileNotFoundError(
        f"No .tsv matching {keywords} found in: {directory}"
    )


def align_genes(*dfs):
    """Retain only genes present in all DataFrames. Returns same number of dfs."""
    common = dfs[0].columns
    for df in dfs[1:]:
        common = common.intersection(df.columns)
    print(f"  Common genes: {len(common)}")
    return tuple(df[common] for df in dfs)


def filter_to_genes(df, genes):
    """Reindex df to saved gene list. Missing genes are filled with NaN."""
    missing = len(set(genes) - set(df.columns))
    if missing:
        print(f"  Warning: {missing} genes missing in test data → filled with NaN")
    return df.reindex(columns=genes)
