import numpy as np
from preprocessing import data_of_jet_num, clean_data
from features import *

### PREDICTION + ACCURACY

def predict(weights, X):
    """
    Returns a list of labels (rounded to -1 or 1) given weights and data X
    """
    labels = np.dot(X, weights)
    labels = np.sign(labels)*1.0
    return labels
    
def predict_all(X, weights, degs, medians, unwanted):
    """
    Predicts all labels given raw data.
    This means it also separates, cleans and expands the data.
    """
    predictions = np.ones((X.shape[0],))
    # for every jet number
    for i in range(4):
        # get test data
        x, ids = data_of_jet_num(X, None, i)  
        # clean test data
        clean_x = clean_data(x, medians[i], unwanted[i])
        expanded_x = add_features(clean_x, degs[i])
        # predict the labels
        p = predict(weights[i], expanded_x) 
        predictions[ids] = p
    return predictions
    
def accuracy(predictions, actual):
    """
    Returns the accuracy given predictions and the actual values
    Accuracy is in range [0, 1]
    """
    return np.sum(predictions==actual)/len(actual)

"""    
def check(test_x, test_y, weights, degs):
    predictions = predict_all(test_x, weights, degs)
    return accuracy(predictions, test_y)
"""

# ----------- Predict -------------
def sigmoid(t):
    """ Sigmoid function
    """
    return 1/(1 + np.exp(-t))

def logistic_regression_classify(data, w):
    """ Make predictions for logistic regression
    """
    predictions = sigmoid(data@w)
    predictions[predictions<0.5]=-1
    predictions[predictions>=0.5]=1        
    return predictions