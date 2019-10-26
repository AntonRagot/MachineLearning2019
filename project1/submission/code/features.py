import numpy as np

### FEATURE ADDING

def expand_poly(X, degree):
    """
    Expands the array X up to the given degree
    e.g. expand_poly([x], 3) gives [1, x, x^2, x^3]
    """
    # add bias term (1)
    to_concatenate = [np.ones((X.shape[0],1)), X.copy()]
    for d in range(2,degree + 1):
        to_concatenate.append(np.power(X, d))
    return np.concatenate(to_concatenate, axis=1)

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
    to_concatenate = [expanded_x]
    for i in range(to):
        for j in range(i+1, to):
            to_concatenate.append((X[:,i] * X[:,j]).reshape(-1, 1)) 
            
    # add cos and sin terms
    to_concatenate.append(np.cos(X))
    to_concatenate.append(np.sin(X))
    
    return np.concatenate(to_concatenate, axis=1)