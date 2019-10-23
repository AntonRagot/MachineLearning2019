# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
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

            
def split_data(x, y, train_ratio, seed=0):
    """
    Split the dataset into train and test sets
    """
    
    # set seed
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
    medians = []
    for i in range(4):
        x = X[X[:, 22] == i]
        medians.append(np.median(x[:,0]))
    return medians
    
### PREPROCESSING

class Classifier:
    
    UNDEF = -999.
    JET_NUM_INDEX = 22
    K = 10
    
    def __init__(self, UNWANTED, MEDIANS, LAMBDAS, DEGREES):
        self.UNWANTED = UNWANTED
        self.MEDIANS = MEDIANS
        self.LAMBDAS = LAMBDAS
        self.DEGREES = DEGREES
    
    def data_of_jet_num(self, X, y, jet_num):
        mask = X[:, self.JET_NUM_INDEX] == jet_num
        if y is None:
            return X[mask], mask
        return X[mask], y[mask], mask
        
    def replace_undef_by_median(self,X,median):
        X_ = X.copy()
        X_[X[:,0] == self.UNDEF, 0] = median
        return X_
        
    def remove_unwanted_cols(self,X,unwanted_cols):
        x_clean = np.delete(X, unwanted_cols, axis=1)
        return x_clean
        
    def clean_data(self,X,median,unwanted_cols):
        X_ = self.remove_unwanted_cols(X,unwanted_cols)
        X_ = self.replace_undef_by_median(X_, median)
        return X_
        
    def clean_data_for_jet_num(self,x,y,jet_num):
        x, y, _ = self.data_of_jet_num(x, y, jet_num)
        x = self.clean_data(x, self.MEDIANS[jet_num], self.UNWANTED[jet_num])
        return x, y
        
    ### EXPANDING

    def expand_poly(self,X, degree):
        expanded_x = X.copy()
        for d in range(2,degree + 1):
                expanded_x = np.c_[expanded_x, np.power(X, d)]
        return expanded_x
        
    ### TRAINING

    def regress(self,x, y, lamb=0):
        w,_ = ridge_regression(y, x, lamb)
        return w
        
    ### PREDICTION + ACCURACY

    def predict(self, w, test_x):
        labels = np.dot(test_x, w)
        labels = np.sign(labels)*1.0
        return labels
        
    def predict_all(self, test_x, weights, degs):
        predictions = np.ones((test_x.shape[0],))
        # for every jet number
        for i in range(4):
            # get test data
            x, ids = self.data_of_jet_num(test_x, None, i)  
            # clean test data (only remove undefined columns)
            clean_x = self.clean_data(x, self.MEDIANS[i], self.UNWANTED[i])
            expanded_x = self.expand_poly(clean_x, degs[i])
            # predict the labels
            p = self.predict(weights[i], expanded_x) 
            predictions[ids] = p
        return predictions
        
    def accuracy(self,predictions, actual):
        return np.sum(predictions==actual)/len(actual)
        
    def check(self,test_x, test_y, weights, degs):
        predictions = self.predict_all(test_x, weights, degs)
        return self.accuracy(predictions, test_y)
        
    ### CROSS-VALIDATION

    def build_k_indices(self, y, seed=0):
        np.random.seed(seed)
        indices = np.random.permutation(len(y))
        k_indices = np.array_split(indices, self.K)
        out = []
        for k in range(self.K):
            te = k_indices[k]
            tr = np.concatenate(np.delete(k_indices, k))
            out.append((tr,te))
        return out
        
    def k_fold_cross_validation(self,x,y):
        accuracies = np.zeros((len(self.DEGREES), len(self.LAMBDAS)))
        k_indices = self.build_k_indices(y)
        c = 1
        for d, degree in enumerate(self.DEGREES):     
            x_ = self.expand_poly(x, degree)
            for l, lamb in enumerate(self.LAMBDAS):  
                temp_acc = []
                for indices in k_indices:
                    tr_ind, te_ind = indices
                    tr_x, tr_y = x_[tr_ind], y[tr_ind]
                    te_x, te_y = x_[te_ind], y[te_ind]
                    w = self.regress(tr_x, tr_y, lamb)
                    preds = self.predict(w, te_x)
                    temp_acc.append(self.accuracy(preds, te_y))   
                accuracies[d][l] = np.mean(temp_acc)
                print('\r%d / %d' % (c, len(self.LAMBDAS)*len(self.DEGREES)), end='')
                c += 1
        best_d, best_l = np.unravel_index(np.argmax(accuracies), accuracies.shape)
        best_d = self.DEGREES[best_d]
        best_l = self.LAMBDAS[best_l]
        print('\n')
        return accuracies, best_d, best_l