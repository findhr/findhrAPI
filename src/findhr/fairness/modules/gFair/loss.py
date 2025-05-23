import numpy as np
from scipy import linalg
from itertools import chain
import os
from ..probabilistic_mapping_helpers import compute_X_hat, dist_pairwise_group

def compute_reconstruction_loss(x_orig, x_hat):
    """Computes the reconstruction loss (L_x)."""
    return np.mean((x_hat - x_orig) ** 2)

def compute_Ligf_group(x_orig, x_hat):
    """Computes the in-group-fairness loss for a single group."""
    D_X_orig = dist_pairwise_group(x_orig, x_orig, np.ones(x_orig.shape[1]))
    D_X_hat = dist_pairwise_group(x_hat, x_hat, np.ones(x_hat.shape[1]))
    return linalg.norm(D_X_orig - D_X_hat)


def compute_Ligf(X, X_hat, sensitive_groups, sensitive_indexes):
    """
    Computes the in-group-fairness (IGF) loss.
    """

    # Combine privileged and unprivileged groups into a single set
    groups = set(chain.from_iterable(
        list(sensitive_groups["privileged_groups"].values()) +
        list(sensitive_groups["unprivileged_groups"].values())
    ))

    # Identify non-sensitive feature indexes
    non_sensitive_indexes = [i for i in range(X.shape[1]) if i not in sensitive_indexes]

    L = []
    for group in groups:
        for s_index in sensitive_indexes:
            # Get mask for rows belonging to the current group
            mask = np.isin(X[:, s_index], group)

            if np.any(mask):  # Proceed only if group members exist
                x_orig = X[mask][:, non_sensitive_indexes]
                x_hat = X_hat[mask][:, non_sensitive_indexes]
                L.append(compute_Ligf_group(x_orig, x_hat))
    return np.sum(L)

def compute_gFair_fairness_loss_pairwise(D_X_f_s1_s2, D_X_f_hat_s1_s2):
    """Computes the group fairness loss based on distances between two groups."""
    group_gap = linalg.norm(D_X_f_s1_s2 - D_X_f_hat_s1_s2)
    return group_gap


def compute_gFair_group_fairness_loss(X, X_hat, params, sensitive_groups, sensitive_indexes, group_weights, D_X_f, biggest_gap=False):
    """Computes the group fairness loss based on distances between the groups."""
    L_z = 0
    N, M = X.shape
    alpha = params[:M]
    if biggest_gap:
        max_group_avg_difference = 0
    for s_attr in sensitive_groups["privileged_groups"].keys():
        privileged_groups = sensitive_groups["privileged_groups"][s_attr]
        unprivileged_groups = sensitive_groups["unprivileged_groups"][s_attr]
        for index_advantaged_group in range(len(privileged_groups)):
            for s_index in sensitive_indexes:
                mask_advantaged = np.isin(element=X[:, s_index],
                                          test_elements=privileged_groups[index_advantaged_group]).reshape(-1)
                if sum(mask_advantaged) != 0:
                    for index_sensitive_group in range(index_advantaged_group + 1, len(unprivileged_groups)):
                        if unprivileged_groups[index_sensitive_group] != privileged_groups[index_advantaged_group]:
                            mask_s = np.isin(element=X[:, s_index],
                                             test_elements=unprivileged_groups[index_sensitive_group]).reshape(-1)
                            if sum(mask_s) != 0:
                                if unprivileged_groups[index_sensitive_group] != privileged_groups[
                                    index_advantaged_group]:
                                    weight = abs(
                                        group_weights[privileged_groups[index_advantaged_group]] - group_weights[
                                            unprivileged_groups[index_sensitive_group]]) + 1
                                    D_X_f_hat_s1_s2 = dist_pairwise_group(X_hat[mask_s], X_hat[mask_advantaged],
                                                                          alpha)
                                    D_X_f_s1_s2 = D_X_f[mask_s][:, mask_advantaged]
                                    if biggest_gap:
                                        if max_group_avg_difference < compute_gFair_fairness_loss_pairwise(D_X_f_s1_s2, D_X_f_hat_s1_s2):
                                            max_group_avg_difference = compute_gFair_fairness_loss_pairwise(D_X_f_s1_s2, D_X_f_hat_s1_s2)
                                            L_z = [weight * compute_gFair_fairness_loss_pairwise(D_X_f_s1_s2, D_X_f_hat_s1_s2)]
                                    else:
                                        L_z += weight * compute_gFair_fairness_loss_pairwise(D_X_f_s1_s2, D_X_f_hat_s1_s2)

    return L_z


def gFair_optimisation(params, X, sensitive_groups, sensitive_indexes, D_X_f, k=10, A_x=1e-4, A_z=1e-4, A_igf=1e-4,
                       group_weights=None, biggest_gap=False, logs_path='', verbose=False, print_interval=100):
    """Optimisation function for gFair.
    Parameters
    ----------
    params :  numpy.ndarray
        params to optimize, in this case this represents the prototypes (params[k:]) and the  w value (params[:k]) - the parameters governing the mapping from prototypes to class predictions (Y).
    X :  numpy.ndarray
        array with size N * M, where N is the number of candidates and M is the number of features for each candidate
    sensitive_groups :  dict
        dict containing the privileged and unprivileged groups of the form {"s_attr" : [value_1, value_2, ...]}
    sensitive_indexes :  list
        index of columns of the dataframe containing the sensitive attributes
    D_X_f :  numpy.ndarray
        pairwise distance matrix between all candidates of size N * N where N is the number of candidates
    k : int
        number of prototypes
    Ax : float
        hyperparameter for Lx, the data loss that should optimize for keeping the new representations as close
        as possible to the original ones
    Az : float
        hyperparameter for Lz, the group fairness loss
    Aigf : float
        hyperparameter for Ligf, the in-group-fairness loss that should ensure that the new representations respect fairness within the group
    group_weights : dict
        dictionary of weights for each group considered in the optimisation
    biggest_gap : bool
        True tto consider during the optimisation only the biggest gap between groups instead of the pairwise sum of the gaps
    logs_path : str
        path to save logs
    verbose : bool
        True to print to console the logs
    print_interval : int
        print/save logs at specified interval
    """

    X_hat, _ = compute_X_hat(X, params, k, alpha=True)

    L_x = compute_reconstruction_loss(X, X_hat)
    L_igf = compute_Ligf(X, X_hat, sensitive_groups, sensitive_indexes)
    L_z = compute_gFair_group_fairness_loss(X, X_hat, params, sensitive_groups, sensitive_indexes, group_weights, D_X_f,
                                            biggest_gap)

    criterion = A_x * L_x + A_z * L_z + A_igf * L_igf
    logs_msg = "step: {}, L_x: {},  L_z: {},  L_igf: {}, loss:{}\n".format(
                    gFair_optimisation.iters, L_x, L_z, L_igf, criterion)
    if gFair_optimisation.iters % print_interval == 0:
        if verbose:
            print(logs_msg)
        if logs_path != '' and logs_path is not None:
            os.makedirs(logs_path, exist_ok=True)
            with open(os.path.join(logs_path, "logs.txt"), 'a') as f:
                f.write(logs_msg)
    gFair_optimisation.iters += 1
    return criterion
