# -*- coding: utf-8 -*-
import csv
import numpy as np
from implementations import ridge_regression

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

    
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

            
def split_data(x, y, train_ratio, seed=0):
    """
    Split the dataset into train and test sets
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

def get_medians(X):
    """
    Returns medians of the DER_mass_MMC feature for each jet num value
    """
    medians = []
    for i in range(4):
        x = X[X[:, 22] == i]
        medians.append(np.median(x[:,0]))
    return medians