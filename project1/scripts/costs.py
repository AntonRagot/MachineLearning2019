# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculates the loss using MSE.
    """
    err = y - tx@w # compute error vector
    return 1.0/(2.0*y.shape[0]) * (err.T @ err)