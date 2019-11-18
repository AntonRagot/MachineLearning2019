import numpy as np
from implementations import ridge_regression
from prediction import accuracy, predict
from features import *
from constants import K

    
### TRAINING

def regress(x, y, lamb=0):
    """
    Computes weights using ridge regression
    """
    w,_ = ridge_regression(y, x, lamb)
    return w
    
### CROSS-VALIDATION

def build_k_indices(y, K=K, seed=0):
    """
    Builds the indices used for k-fold cross validation.
    Returns a list of (train indices, test indices) where test indices is the k-th 'block'
    and train indices are the rest
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(y))
    k_indices = np.array_split(indices, K)
    out = []
    for k in range(K):
        te = k_indices[k]
        tr = np.concatenate(np.delete(k_indices, k))
        out.append((tr,te))
    return out
    
def k_fold_cross_validation(x,y,degrees,lambdas):
    """
    Does a k-fold cross validation to find the best degree and lambda,
    given a list of degrees and lambdas to test.
    Returns all measured accuracies as well as the best degree and lambda.
    """
    accuracies = np.zeros((len(degrees), len(lambdas)))
    k_indices = build_k_indices(y)
    c = 1
    for d, degree in enumerate(degrees):     
        x_ = add_features(x, degree)
        for l, lamb in enumerate(lambdas):  
            temp_acc = []
            for indices in k_indices:
                tr_ind, te_ind = indices
                tr_x, tr_y = x_[tr_ind], y[tr_ind]
                te_x, te_y = x_[te_ind], y[te_ind]
                w = regress(tr_x, tr_y, lamb)
                preds = predict(w, te_x)
                temp_acc.append(accuracy(preds, te_y))   
            accuracies[d][l] = np.mean(temp_acc)
            print('\r%d / %d' % (c, len(lambdas)*len(degrees)), end='')
            c += 1
    best_d, best_l = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    best_d = degrees[best_d]
    best_l = lambdas[best_l]
    print('\n')
    return accuracies, best_d, best_l