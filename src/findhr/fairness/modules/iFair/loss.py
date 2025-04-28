"""
Implementation of the ICDE 2019 paper
iFair_module: Learning Individually Fair Data Representations for Algorithmic Decision Making
__url__: https://ieeexplore.ieee.org/abstract/document/8731591
__author__: Preethi Lahoti
__email__: plahoti@mpi-inf.mpg.de
"""

import numpy as np
import os
from scipy import linalg
from sklearn.metrics import pairwise
from ..probabilistic_mapping_helpers import compute_X_hat


def compute_reconstruction_loss(x_orig, x_hat):
    """Computes the reconstruction loss (L_x)."""
    return np.mean((x_hat - x_orig) ** 2)


def compute_individual_fairness_loss(D_X_f, X_hat):
    """Computes the individual fairness loss (L_z)."""
    D_X_f_hat = pairwise.euclidean_distances(X_hat, X_hat)
    L_z = linalg.norm(D_X_f - D_X_f_hat)
    return L_z


def iFair_optimisation(params, X, D_X_f=None, k=10, A_x=1e-4, A_z=1e-4, logs_path="", verbose=False,
                       print_interval=100):
    """iFair optimisation function.

    Parameters
    ----------
    params :  numpy.ndarray
        params to optimize, in this case this represents the prototypes (params[M:]) and the alpha value (params[:M]) - importance of each feature in the similarity computation
    X :  numpy.ndarray
        array with size N * M, where N is the number of candidates and M is the number of features for each candidate
    D_X_f :  numpy.ndarray
        pairwise distance matrix between all candidates of size N * N where N is the number of candidates
    k : int
        number of prototypes
    Ax : float
        hyperparameter for Lx, the data loss that should optimize for keeping the new representations as close
        as possible to the original ones
    Az : float
        hyperparameter for Lz, the individual fairness loss
    logs_path : str
        path to save logs
    verbose : bool
        True to print to console the logs
    print_interval : int
        print/save logs at specified interval
    """

    X_hat, _ = compute_X_hat(X, params, k, alpha=True)

    L_x = compute_reconstruction_loss(X, X_hat)
    L_z = compute_individual_fairness_loss(D_X_f, X_hat)

    criterion = A_x * L_x + A_z * L_z

    if  iFair_optimisation.iters % print_interval == 0:
        log_msg = "step: {}, L_x: {},  L_z: {}, loss:{}\n".format(
                iFair_optimisation.iters, L_x, L_z, criterion)
        if verbose:
            print(log_msg)
        if logs_path != '' and logs_path is not None:
            os.makedirs(logs_path, exist_ok=True)
            with open(os.path.join(logs_path, "logs.txt"), 'a') as f:
                f.write(log_msg)
    iFair_optimisation.iters += 1

    return criterion
