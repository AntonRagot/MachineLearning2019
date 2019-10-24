import numpy as np

### FEATURE ADDING

def expand_poly(X, degree):
    expanded_x = X.copy()
    for d in range(2,degree + 1):
            expanded_x = np.c_[expanded_x, np.power(X, d)]
    return expanded_x

def add_features(X, degree):
    expanded_x = X.copy()
    
    # Add polynomial terms
    expanded_x = expand_poly(X, degree)
    
    # Add cross terms
    to = X.shape[1]
    for i in range(to):
        for j in range(i+1, to):
            expanded_x = np.c_[expanded_x, X[:,i] * X[:,j]] 
            
    #Add cos and sin terms
    expanded_x = np.c_[expanded_x, np.cos(X)]
    expanded_x = np.c_[expanded_x, np.sin(X)]
    
    return expanded_x