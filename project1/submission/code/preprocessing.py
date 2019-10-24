import numpy as np
from constants import *

### PREPROCESSING

def data_of_jet_num(X, y, jet_num):
    mask = X[:, JET_NUM_INDEX] == jet_num
    if y is None:
        return X[mask], mask
    return X[mask], y[mask], mask
    
def replace_undef_by_median(X,median):
    X_ = X.copy()
    X_[X[:,0] == UNDEF, 0] = median
    return X_
    
def remove_unwanted_cols(X,unwanted_cols):
    x_clean = np.delete(X, unwanted_cols, axis=1)
    return x_clean
    
def clean_data(X,median,unwanted_cols):
    X_ = remove_unwanted_cols(X,unwanted_cols)
    X_ = replace_undef_by_median(X_, median)
    return X_
    
def clean_data_for_jet_num(x,y,jet_num,medians,unwanted):
    x, y, _ = data_of_jet_num(x, y, jet_num)
    x = clean_data(x, medians[jet_num], unwanted[jet_num])
    return x, y