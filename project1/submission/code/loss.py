import numpy as np
from prediction import sigmoid

### LOSS FUNCTIONS

def mse(y, tx, w):
    """Computes loss using mean square error
    """
    err = y - tx @ w # compute error vector
    return 1.0/(2.0*y.shape[0]) * (err.T @ err)
    
def logistic_loss(y, tx, w): 
    """Computes loss using log-likely-hood
    """
    return np.sum(np.log(1+np.exp(np.dot(tx,w))) - y * np.dot(tx,w))

def reg_logistic_loss(y, tx, w, lambda_):
    """Computes loss using regularized log-likely-hood
    """
    return logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)
    
def mae(y, tx, w):
    """Computes loss using mean absolute error
    """
    err = y - tx @ w
    return np.mean(np.abs(err))
    
def rmse(y, tx, w):
    """Computes loss using root mean square error
    """
    return np.sqrt(2 * mse(y, tx, w))

### GRADIENT FUNCTIONS

def compute_gradient(y, tx, w):
    """Compute the gradient
    """
    err = y - tx@w # error vector
    return -1.0/y.shape[0] * tx.T @ err

def mse_grad(y, tx, w):
    """Computes gradient of mse function
    """
    err = y - tx @ w
    return -1.0/y.shape[0] * (tx.T @ err)

def logistic_grad(y, tx, w):
    """Computes gradient of logistic function
    """
    return np.dot(tx.T, sigmoid(np.dot(tx, w))-y)


def reg_logistic_grad(y, tx, w, lambda_):
    """Computes gradient of regularized logistic function
    """
    return logistic_grad(y, tx, w) + 2 * lambda_ * w
