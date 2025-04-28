import os, pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise


def writeToCSV(file_name_with_path, _df):
    """
    Method to generate the fair ranking based on the score column.
    It adds to the dataframe a column indicating the rank of the candidate.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to be saved

    file_name_with_path : str
        path to save the dataframe

    """

    try:
        _df.to_csv(file_name_with_path, index=False)
    except FileNotFoundError:

        directory = os.path.dirname(file_name_with_path)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Make folder ", directory)
        _df.to_csv(file_name_with_path, index=False)


def compute_euclidean_distances(X, nonsensitive_column_indices):
    """
    Method to generate the fair ranking based on the score column.
    It adds to the dataframe a column indicating the rank of the candidate.

    Parameters
    ----------
    X : pandas.DataFrame
        dataframe containing the data for which to compute the pairwise distances

    euclidean_dist_dir : str
        path to save the computed distances

    batch : int
        batch size to compute the distances

    nonsensitive_column_indices : list[int]
        indices indicating the nonsensitive columns to be used for computing the distance

    """
    if not nonsensitive_column_indices:
        nonsensitive_column_indices = list(range(0, X.shape[1] - 1))

    D_X_F = pairwise.euclidean_distances(X[:, nonsensitive_column_indices],
                                         X[:, nonsensitive_column_indices])
    return D_X_F


def generate_ranking(df, query_col, score_col, ascending, rank_col, out_path=None):
    """
    Method to generate the fair ranking based on the score column.
    It adds to the dataframe a column indicating the rank of the candidate.

    Parameters
    ----------
    df : pandas.DataFrame
       dataframe to be ranked

    query_col : str
       name of column representing the query (e.g. job title)

    score_col : str
       name of column representing the score (e.g.  score_fair) to rank the candidates

    ascending : bool
        indicates whether to sort the candidates by score_col in ascending (True) or descending (False) order

    rank_col : str
       name of column to be added in the dataframe that will contain the rank of each candidate

    out_path : str
        path to save the dataframe with the rank column added

    Returns
    -------
    fair_data: pandas.DataFrame
      dataframe containing the fair transformed data representation as col_name + "_fair"
    """
    if out_path is None or not os.path.exists(out_path):
        df_ranked = df.groupby(query_col).apply(
            lambda x: x.sort_values(by=score_col, ascending=ascending)).reset_index(drop=True)
        df_ranked[rank_col] = df_ranked.groupby(query_col).cumcount()
        df_ranked[rank_col] = df_ranked[rank_col] + 1

        if out_path is not None:
            df_ranked.to_csv(out_path)
    else:
        df_ranked = pd.read_csv(out_path)

    return df_ranked


def compute_dynamic_values_to_code(data, sensitive_attributes):
    value_to_code = {}
    current_code = 1

    # For each sensitive attribute
    for s_attribute in sorted(sensitive_attributes):
        unique_groups = sorted(set(data[s_attribute].unique()))

        # Assign code to independent groups
        for group in unique_groups:
            if group not in value_to_code:
                value_to_code[group] = current_code
                current_code += 1

    return value_to_code


def process_data_input(data, configs):
    """

    Parameters
    ----------
    data
    configs

    Returns
    -------
    sensitive_groups :  dict
            dict containing the privileged and unprivileged groups of the form
            {
                "unprivileged_groups" : {"s_attr" : [value_1, value_2, ...]},
                "privileged_groups" : {"s_attr" : [value_1, value_2, ...]}
            }

            for the optimisation of intersectional groups
    """
    features_col = configs["feature_cols"] + [configs["score_col"]]

    if "optimisation_type" in configs:
        configs["sensitive_groups"] = get_sensitive_groups(data, configs["optimisation_type"], configs["sensitive_col"])
        sensitive_groups_configs = configs["sensitive_groups"]
        sensitive_attributes = set(sensitive_groups_configs["unprivileged_groups"]) | set(
            sensitive_groups_configs["privileged_groups"])
    else:
        sensitive_attributes = configs["sensitive_col"]
    value_to_code = compute_dynamic_values_to_code(data, sensitive_attributes)

    unprivileged_groups, privileged_groups, group_weights = {}, {}, {}
    # Encode sensitive attributes
    for s_attribute in sensitive_attributes:
        coded_attribute = s_attribute + '_coded'
        features_col.append(coded_attribute)

        data.loc[:, coded_attribute] = data[s_attribute].map(value_to_code)
        if 'sensitive_groups' in configs:
            unprivileged_groups[s_attribute] = [value_to_code[group] for group in
                                                sensitive_groups_configs["unprivileged_groups"].get(s_attribute, [])]
            privileged_groups[s_attribute] = [value_to_code[group] for group in
                                              sensitive_groups_configs["privileged_groups"].get(s_attribute, [])]

        if 'sensitive_groups' in configs:
            # Update group weights
            for group in value_to_code:
                if group in data[s_attribute].unique() and value_to_code[group] not in group_weights:
                    if configs['group_weights'] is not None:
                        group_weights[value_to_code[group]] = configs['group_weights'].get(group, 1)
                    else:
                        group_weights[value_to_code[group]] = 1

    if "sensitive_groups" in configs:
        # Store sensitive groups
        sensitive_groups = {"unprivileged_groups": unprivileged_groups, "privileged_groups": privileged_groups}

    # Identify indices of sensitive columns
    sensitive_column_indices = [
        list(data[features_col].columns).index(s_attribute + "_coded") for s_attribute in sensitive_attributes
    ]

    nonsensitive_column_indices = [index for index in range(0, len(features_col)) if
                                   index not in sensitive_column_indices]
    data_processed = data[features_col].to_numpy()
    if "sensitive_groups" in configs:
        return data_processed, group_weights, sensitive_groups, sensitive_column_indices, nonsensitive_column_indices
    else:
        return data_processed, sensitive_column_indices, nonsensitive_column_indices


def get_sensitive_groups(data, optimisation_config, sensitive_cols):
    optimisation_type = list(optimisation_config.keys())[0]
    optimisation_values = optimisation_config[optimisation_type]

    intersectional_opt = ["pairwise", "control", "extremes"]
    all_opt = ["independent"] + intersectional_opt
    if optimisation_type not in all_opt:
        raise Exception("Optimisation type not supported!")

    if optimisation_type in intersectional_opt:
        column_name = []
        temp = [""] * len(data)
        for col in sensitive_cols:
            column_name.append(col)
            temp = temp + data[col].astype(str)
        sensitive_col = "_".join(column_name)
        data.loc[:, sensitive_col] = temp
        groups = list(data[sensitive_col].unique())

        if optimisation_type == "pairwise":
            sensitive_groups = {
                "unprivileged_groups": {sensitive_col: groups},
                "privileged_groups": {sensitive_col: groups},
            }
        elif optimisation_type == "control":
            sensitive_groups = {
                "unprivileged_groups": {sensitive_col: groups},
                "privileged_groups": {sensitive_col: [optimisation_values]},
            }
        elif optimisation_type == "extremes":
            sensitive_groups = {
                "unprivileged_groups": {sensitive_col: optimisation_values},
                "privileged_groups": {sensitive_col: [optimisation_values[0]]},
            }
    elif optimisation_type == "independent":
        sensitive_groups = {"privileged_groups": {}, "unprivileged_groups": {}}

        for col in sensitive_cols:
            groups = data[col].unique()
            sensitive_groups["privileged_groups"][col] = groups
            sensitive_groups["unprivileged_groups"][col] = groups

    return sensitive_groups


def process_data_output(data_orig, data_fair, configs, nonsensitive_column_indices, fair_data_path, file_name=None):
    # Combine relevant columns from configs
    features_col = configs["feature_cols"] + [configs["score_col"]]

    # Convert data_fair to a numpy array and select relevant columns
    data_fair = np.vstack(data_fair)
    fair_features_col = [f'{col}_fair' for col in features_col]
    selected_fair_features_col = [fair_features_col[i] for i in nonsensitive_column_indices]

    # Create DataFrame for fair data with selected columns
    data_fair = pd.DataFrame(data_fair[:, nonsensitive_column_indices], columns=selected_fair_features_col)

    # Add original data columns to the fair data
    for col in data_orig.columns:
        data_fair[col] = data_orig[col].values
        data_fair[col] = data_orig[col].values

    if file_name != None:
        os.makedirs(fair_data_path, exist_ok=True)
        data_fair.to_csv(os.path.join(fair_data_path, f'fair_{file_name}_data.csv'))

    return data_fair


def save_model_data(model, path, qid):
    with open(os.path.join(path, f'model_parmas_{qid}.npy'), 'wb') as f:
        np.save(f, model.opt_params[qid])


def load_model_data(path, qid):

    with open(os.path.join(path, f'model_parmas_{qid}.npy'), 'rb') as f:
        return np.load(f, allow_pickle=True)
