import arff  # liac-arff
import numpy as np
import pandas as pd

def _load_arff_to_df(filepath):
    """Helper function to load an ARFF file into a Pandas DataFrame."""
    with open(filepath, 'r') as f:
        dataset = arff.load(f)
    
    data = np.array(dataset['data'])
    attributes = [attr[0] for attr in dataset['attributes']]
    return pd.DataFrame(data, columns=attributes)

def load_aggregation(filepath):
    print("Loading Aggregation Dataset...")
    df = _load_arff_to_df(filepath)
    X = df[['x', 'y']].values
    y = df['class'].values
    return X, y

def load_cluto(filepath):
    print("Loading Cluto-t4-8k Dataset...")
    df = _load_arff_to_df(filepath)
    
    # Handle the target column name and filter out 'noise'
    df['class'] = df['CLASS'].astype(str)
    df = df[df['class'] != 'noise'].copy()
    
    X = df[['x', 'y']].values
    y = df['class'].values
    return X, y

def load_dartboard(filepath):
    print("Loading Dartboard Dataset...")
    df = _load_arff_to_df(filepath)
    X = df[['a0', 'a1']].values  # Note: Features are a0, a1 here
    y = df['class'].values
    return X, y

def load_pathbased(filepath):
    print("Loading Pathbased Dataset...")
    df = _load_arff_to_df(filepath)
    X = df[['x', 'y']].values
    y = df['class'].values
    return X, y