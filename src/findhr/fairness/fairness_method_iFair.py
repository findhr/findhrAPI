
"""
Implementation of the ICDE 2019 paper
iFair_module: Learning Individually Fair Data Representations for Algorithmic Decision Making
url: https://ieeexplore.ieee.org/document/8731591
citation:
Lahoti, P., Gummadi, K. P., & Weikum, G. (2019, April).
ifair: Learning individually fair data representations for algorithmic decision making.
In 2019 ieee 35th international conference on data engineering (icde) (pp. 1334-1345). IEEE.

__author__: Preethi Lahoti
__email__: plahoti@mpi-inf.mpg.de
"""

import os
import numpy as np
import pandas as pd
from findhr.fairness.modules.iFair.helpers import train, transform
from findhr.fairness.utils.utils import save_model_data, load_model_data
from findhr.fairness.fairness_method import PreprocessingFairnessIntervention

class iFair(PreprocessingFairnessIntervention):
    """
    Class for applying the iFair pre-processing fairness intervention.

    Attributes
    ----------
    query_col : str
        name of column representing the query (e.g. job title, job category)
    sensitive_col :  list[str]
        column containing the sensitive attributes
    feature_cols : list[str]
        list of columns representing the features to be transformed
    score_col : str
        name of column representing the score given by the system
    k : int
        number of prototypes
    A_x : float
        hyperparameter for Lx, the data loss that should optimize for keeping the new representations as close
        as possible to the original ones
    A_z : float
        hyperparameter for Lz, the individual fairness loss
    model_occ : bool
        true if for optimizing for each query separately, false otherwise
    pos_th : int
        score threshold for using only candidates with the score above pos_th for training the pre-processing method
    maxfun : int
        maxinum number of function evaluations.
    maxiter : int
        maximum number of iterations.
    nb_restarts : int
        number of repetitions
    verbose : bool
        True to print to console the logs
    print_interval : int
        print/save logs at specified interval
    out_path : str
        path to save the model, logs and fair data representation
    file_name : str
            file_name to save the transformed data
    seed : int
            set seed
    """
    def __init__(self, query_col, sensitive_col, feature_cols, score_col, k, A_x, A_z,
                 model_occ=False, pos_th=None,  maxfun=1000, maxiter=1000, nb_restarts=3,
                 verbose=False, print_interval=100,  out_path=None, file_name=None, seed=42):
        """
            Initialize the iFair model with the params and creates a configuration dict.
        """
        configs = {key: value for key, value in locals().items() if key != "self"}
        configs["name"] = "iFair"
        super().__init__(configs)

    def fit(self, data, y=None):
        """
        Trains the pre-processing fairness intervention model on data.

        Parameters
        ----------
        data : pandas.DataFrame
            dataframe to be used for training;
            it needs to contain the sensitive information

        """

        if self.configs["model_occ"]:
            qids = data[self.configs["query_col"]].unique()
        else:
            qids = ["all"]

        # if pos_th is not set, consider all data
        if self.configs["pos_th"] is None:
            self.configs["pos_th"] = min(data[self.configs["score_col"]]) - 0.01
        data_positive = data[data[self.configs["score_col"]] > self.configs["pos_th"]]

        if self.is_start_train():
            if self.configs["seed"] is not None:
                np.random.seed(self.configs["seed"])

            for qid in qids:
                if self.configs["model_occ"]:
                    data_qid = data_positive[data_positive[self.configs["query_col"]] == qid]
                else:
                    data_qid = data_positive

                self.opt_params[qid] = train(data_qid, self.configs, self.model_path)

                if self.configs["out_path"] is not None:
                    save_model_data(self, os.path.join(self.model_path), qid)

    def transform(self, data):
        """
        Using the pre-processing fairness model it generates the fair data representation for data.

        Parameters
        ----------
        data : pandas.DataFrame
           dataframe for which to generate the fair data representation;
           it does not need to contain the sensitive information

        Returns
        -------
        X_fair: pandas.DataFrame
          dataframe containing the fair transformed data representation as col_name + "_fair"
        """
        if self.configs["model_occ"]:
            qids = data[self.configs["query_col"]].unique()
        else:
            qids = ["all"]

        if self.is_transform():
            data_fair = []
            for qid in qids:
                # init opt_params if None from model_path
                if self.opt_params == {} or self.opt_params is None:
                    self.opt_params[qid] = load_model_data(self.model_path, qid)

                if self.configs["model_occ"]:
                    data_qid = data[data[self.configs["query_col"]] == qid]
                else:
                    data_qid = data
                X_fair_qid = transform(data_qid, self.opt_params[qid], self.configs)
                data_fair.append(X_fair_qid)
            data_fair = pd.concat(data_fair)
            if self.configs["out_path"] is not None and self.configs["file_name"] is not None:
                fair_data_file = os.path.join(self.fair_data_path, f'fair_{self.configs["file_name"]}_data.csv')
                data_fair.to_csv(fair_data_file)
        else:
            if self.configs["out_path"] is not None and self.configs["file_name"] is not None:
                fair_data_file = os.path.join(self.fair_data_path, f'fair_{self.configs["file_name"]}_data.csv')
                data_fair = pd.read_csv(fair_data_file)
        return data_fair




