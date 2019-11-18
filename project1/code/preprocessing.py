import numpy as np
from constants import *

### DATA SEPARATION

def split_data(x, y, train_ratio, seed=0):
    """
    Splits the dataset into train and test sets
    """
    np.random.seed(seed)
    
    n = y.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    limit = int(n * train_ratio)
    
    train_ids, test_ids = indices[:limit], indices[limit:]
    
    train_x, test_x = x[train_ids], x[test_ids]
    train_y, test_y = y[train_ids], y[test_ids]
    return train_x, train_y, test_x, test_y

def data_of_jet_num(X, y, jet_num):
    """
    Returns subset of data (as well as its indices as a mask) where the jet num is equal to jet_mum
    """
    mask = X[:, JET_NUM_INDEX] == jet_num
    if y is None:
        return X[mask], mask
    return X[mask], y[mask], mask
    
def get_medians(X):
    """
    Returns medians of the DER_mass_MMC feature for each jet num value
    """
    medians = []
    for i in range(4):
        x = X[X[:, JET_NUM_INDEX] == i]
        medians.append(np.median(x[:,0]))
    return medians
    
### CLEANING
    
def replace_undef_by_median(X,median):
    """
    Replaces all undefined values in the first feature (DER_mass_RAP) by the median value
    """
    X_ = X.copy()
    X_[X[:,0] == UNDEF, 0] = median
    return X_
    
def remove_unwanted_cols(X,unwanted_cols):
    """
    Removes unwanted columns (features)
    """
    x_clean = np.delete(X, unwanted_cols, axis=1)
    return x_clean
    
def clean_data(X,median,unwanted_cols):
    """
    Removes unwanted columns and replaces undefined values by median values
    (See remove_unwanted_cols and replace_undef_by_median)
    """
    X_ = remove_unwanted_cols(X,unwanted_cols)
    X_ = replace_undef_by_median(X_, median)
    return X_
    
def clean_data_for_jet_num(x,y,jet_num,medians,unwanted):
    """
    Gets subset of data where the jet num values is jet_num and cleans the data
    (See data_of_jet_num and clean_data)
    """
    x, y, _ = data_of_jet_num(x, y, jet_num)
    x = clean_data(x, medians[jet_num], unwanted[jet_num])
    return x, y