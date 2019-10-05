import numpy as np

# -- Loss functions -----------------------------------------

def mse(y, tx, w):
    """Computes loss using mean square error
    """
    err = y - tx @ w # compute error vector
    return 1.0/(2.0*y.shape[0]) * (err.T @ err)
    
def mse_grad(y, tx, w):
    """Computes gradient of mse function
    """
    err = y - tx @ w
    return grad = -1.0/y.shape[0] * (tx.T @ err)
    
def mae(y, tx, w):
    """Computes loss using mean absolute error
    """
    err = y - tx @ w
    return np.mean(np.abs(err))
    
def rmse(y, tx, w):
    """Computes loss using root mean square error
    """
    return np.sqrt(2 * mse(y, tx, w))
    
# -- Regression functions -----------------------------------
    
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

def logistic_regression(y, tx, initial_w, max_iters, gamma, loss_function=mse, gradient=mse_grad):
    """Logistic regression using SGD
    """
    # TODO
    return w, loss
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, loss_function=mse, gradient=mse_grad):
    """Regularized logistic regression using SGD
    """
    # TODO
    return w, loss