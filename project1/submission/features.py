import numpy as np

### FEATURE ADDING

def expand_poly(X, degree):
    expanded_x = X.copy()
    for d in range(2,degree + 1):
            expanded_x = np.c_[expanded_x, np.power(X, d)]
    return expanded_x
    
def add_features(X, degree):
    # TODO add pairwise and cos/sin
    return expand_poly(X, degree)