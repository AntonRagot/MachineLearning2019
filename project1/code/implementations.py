import numpy as np
from loss import *

### REGRESSION FUNCTIONS
    
def least_squares_GD(y, tx, initial_w, max_iters, gamma, loss_function=mse, gradient=mse_grad):
    """Regression using gradient descent
    """
    w = initial_w
    for iter in range(max_iters):
        # compute gradient
        grad = gradient(y, tx, w)
        # update w
        w = w - gamma * grad
    loss = loss_function(y, tx, w)
    return w, loss
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_function=mse, gradient=mse_grad):
    """Regression using stochastic gradient descent
    """
    w = initial_w
    for iter in range(max_iters):
        # randomly select datapoint
        id = np.random.randint(y.shape[0])
        sample_y, sample_x = y[id], tx[id]
        # compute gradient
        grad = gradient(y, tx, w)
        # update w
        w = w - gamma * grad
    loss = loss_function(y, tx, w)
    return w, loss
    
def least_squares(y, tx, loss_function=rmse):
    """Linear regression using normal equations
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = loss_function(y, tx, w)
    return w, loss
    
def ridge_regression(y, tx, lambda_, loss_function=rmse):
    """Ridge regression using normal equations
    """
    lamb = 2 * y.shape[0] * lambda_
    w = np.linalg.solve(tx.T @ tx + lamb * np.identity(tx.shape[1]), tx.T @ y)
    loss = loss_function(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using SGD
    """
    return least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_function=logistic_loss, gradient=logistic_grad)
    
def reg_logistic_regression(y, tx, lambdas, initial_w, max_iters, gamma):
    """Regularized logistic regression using SGD
    """
    w = initial_w
    for iter in range(max_iters):
        # compute gradient
        grad = reg_logistic_grad(y, tx, w, lambdas)
        # update w
        w = w - gamma * grad
    loss = reg_logistic_loss(y, tx, w, lambdas)
    return w, loss