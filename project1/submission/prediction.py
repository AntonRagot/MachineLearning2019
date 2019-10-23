import numpy as np
from preprocessing import data_of_jet_num, clean_data
from features import *

### PREDICTION + ACCURACY

def predict( w, test_x):
    labels = np.dot(test_x, w)
    labels = np.sign(labels)*1.0
    return labels
    
def predict_all( test_x, weights, degs, medians, unwanted):
    predictions = np.ones((test_x.shape[0],))
    # for every jet number
    for i in range(4):
        # get test data
        x, ids = data_of_jet_num(test_x, None, i)  
        # clean test data
        clean_x = clean_data(x, medians[i], unwanted[i])
        expanded_x = add_features(clean_x, degs[i])
        # predict the labels
        p = predict(weights[i], expanded_x) 
        predictions[ids] = p
    return predictions
    
def accuracy(predictions, actual):
    return np.sum(predictions==actual)/len(actual)
    
def check(test_x, test_y, weights, degs):
    predictions = predict_all(test_x, weights, degs)
    return accuracy(predictions, test_y)