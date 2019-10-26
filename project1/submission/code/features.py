import numpy as np

### FEATURE ADDING

def expand_poly(X, degree):
    """
    Expands the array X up to the given degree
    e.g. expand_poly([x], 3) gives [1, x, x^2, x^3]
    """
    expanded_x = X.copy()
    for d in range(2,degree + 1):
        expanded_x = np.c_[expanded_x, np.power(X, d)]
    # add bias term (1)
    expanded_x = np.c_[np.ones((X.shape[0],1)), expanded_x]
    return expanded_x

def add_features(X, degree):
    """
    Augments the features of X
    Adds polynomial terms (see expand_poly), cross terms and cos/sin of the features
    """
    expanded_x = X.copy()
    
    # add polynomial terms
    expanded_x = expand_poly(X, degree)
    
    # add cross terms
    to = X.shape[1]
    for i in range(to):
        for j in range(i+1, to):
            expanded_x = np.c_[expanded_x, X[:,i] * X[:,j]] 
            
    # add cos and sin terms
    expanded_x = np.c_[expanded_x, np.cos(X)]
    expanded_x = np.c_[expanded_x, np.sin(X)]
    
    return expanded_x