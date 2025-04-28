from scipy.optimize import minimize
import numpy as np
from findhr.fairness.modules.gFair.loss import gFair_optimisation as gfair_func
from findhr.fairness.modules.probabilistic_mapping_helpers import compute_X_hat, compute_euclidean_distances
from findhr.fairness.utils.utils import process_data_input, process_data_output


def train(data, configs, model_path):
    X_train, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices = (
        process_data_input(data, configs))

    indices = list(range(0, X_train.shape[1]))
    nonsensitive_column_indices = [index for index in indices if index not in sensitive_column_indices]
    D_X_F = compute_euclidean_distances(X_train, nonsensitive_column_indices)
    l = len(nonsensitive_column_indices)

    P = X_train.shape[1]
    min_obj = None
    opt_params = None

    for i in range(configs["nb_restarts"]):
        x0_init = np.random.uniform(size=int(P + P * configs["k"]))
        # setting protected column weights to epsilon
        ## assumes that the column indices from l through P are protected and appear at the end
        for i in range(l, P, 1):
            x0_init[i] = 0.0001
        bnd = [(0, 1)] * P + [(None, None)] * P * configs["k"]

        gfair_func.iters = 0
        opt_result = minimize(gfair_func, x0_init,
                              args=(X_train, sensitive_groups, sensitive_column_indices, D_X_F,
                                    configs["k"], configs["A_x"], configs["A_z"],
                                    configs["A_igf"], group_weights, configs["biggest_gap"], model_path,
                                    configs["verbose"], configs["print_interval"]),
                              method='L-BFGS-B',
                              jac=False,
                              bounds=bnd,
                              options={'maxiter': configs["maxiter"],
                                       'maxfun': configs["maxfun"],
                                       'eps': 1e-3})

        if (min_obj is None) or (opt_result.fun < min_obj):
            min_obj = opt_result.fun
            opt_params = opt_result.x
    return opt_params


def transform(data, opt_params, configs):
    X_np, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices = (
        process_data_input(data, configs))

    X_hat, _ = compute_X_hat(X_np, opt_params, configs["k"], alpha=True)
    X_fair = process_data_output(data, X_hat, configs, nonsensitive_column_indices, None, None)
    return X_fair
