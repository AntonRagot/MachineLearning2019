# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


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


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


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

            
def split_data(x, y, train_ratio, validation_ratio, seed=0):
    """
    Split the dataset into train, validation and train sets based on the ratios.
    The sum of the ratios should be < 1.
    """
    if train_ratio + validation_ratio > 1:
        print('Sum of ratio\'s > 1 !')
        return
    
    # set seed
    np.random.seed(seed)
    
    n = y.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    limit_1 = int(n * train_ratio)
    limit_2 = int(n * (train_ratio + validation_ratio))
    
    train_ids, validation_ids, test_ids = indices[:limit_1], indices[limit_1:limit_2], indices[limit_2:]
    
    train_x, validation_x, test_x = x[train_ids], x[validation_ids], x[test_ids]
    train_y, validation_y, test_y = y[train_ids], y[validation_ids], y[test_ids]
    return train_x, train_y, validation_x, validation_y, test_x, test_y
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    return np.vander(x, N=degree+1, increasing=True)