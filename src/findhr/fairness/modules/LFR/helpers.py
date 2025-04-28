import numpy as np
from scipy.optimize import minimize
from findhr.fairness.utils.utils import process_data_input, process_data_output
from findhr.fairness.modules.LFR.loss import LFR_optimisation as LFR_func
from findhr.fairness.modules.probabilistic_mapping_helpers import compute_X_hat


def train(data, configs, model_path):
    X_train, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices = (
        process_data_input(data, configs))


    # assumes that the last non-sensitive column of X_train is Y_train
    Y_train = (X_train[:, nonsensitive_column_indices][:, -1] > configs["pos_th"]).astype(int)

    features_dim = X_train.shape[1]
    min_obj = None
    opt_params = None

    for i in range(configs["nb_restarts"]):
        # Initialize the LFR_module optim objective parameters
        parameters_initialization = np.random.uniform(
            size=int(configs["k"] + features_dim * configs["k"]))

        bnd = [(0, 1)] * configs["k"] + [(None, None)] * features_dim * configs["k"]
        LFR_func.steps = 0

        opt_result = minimize(
            LFR_func, x0=parameters_initialization,
            args=(X_train, Y_train, sensitive_groups, sensitive_column_indices, configs["k"],
                  configs["A_x"], configs["A_y"], configs["A_z"],
                  group_weights, configs["biggest_gap"], model_path, configs["verbose"], configs["print_interval"]),
            method='L-BFGS-B',
            jac=False,  # equivalent to approx_grad=True
            bounds=bnd,
            options={'maxiter': configs["maxiter"], 'maxfun': configs["maxfun"], 'eps': 1e-3}
        )

        if (min_obj is None) or (opt_result.fun < min_obj):
            min_obj = opt_result.fun
            opt_params = opt_result.x

    return opt_params


def transform(data, opt_params, configs):
    X_np, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices = (
        process_data_input(data, configs))
    X_hat, _ = compute_X_hat(X_np, opt_params, configs["k"], alpha=False)
    X_fair = process_data_output(data, X_hat, configs, nonsensitive_column_indices, None, None)
    return X_fair

