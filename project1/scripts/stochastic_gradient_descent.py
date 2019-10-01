# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx@w # error vector
    return -1.0/y.shape[0] * tx.T@err


# /!\ FAUT CHECKER CTE MERDE /!\
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # fetch a batch
        batch = batch_iter(y, tx, batch_size)
        grad = 0
        for y_batch, tx_batch in batch:
            # compute grad of current batch
            grad += compute_stoch_gradient(y_batch, tx_batch, w)
        
        # compute loss
        loss = compute_loss(y, tx, w)
        
        # update weights
        w = w - gamma*grad
        
        ws.append(w)
        losses.append(loss)
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, ws