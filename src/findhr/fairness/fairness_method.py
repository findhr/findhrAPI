import os
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessingFairnessIntervention(TransformerMixin, BaseEstimator):
    """
    Abstract base class for preprocessing fairness interventions.

    This class serves as a base for various fairness intervention methods that preprocess datasets
    to reduce biases while maintaining data utility.
    """

    def __init__(self, configs):
        """
        Initialize the fairness intervention with the given configurations and dataset.

        This method sets up the necessary directories for storing model parameters and
        fair data transformations.

        :param configs : dict
            Configuration dictionary containing model parameters and paths.

        Attributes:
        - self.configs : dict
            Stores the configuration parameters.
        - self.model_path : Path
            Directory path where model parameters will be stored.
        - self.fair_data_path : Path
            Directory path where transformed fair data will be stored.

        Note:
        - The necessary directories are automatically created if they do not exist.
        """
        self.configs = configs
        self.opt_params = dict()
        if configs["out_path"] is not None:
            self.model_path = Path(configs["out_path"]) / self.configs["name"] / "model_params"
            os.makedirs(self.model_path, exist_ok=True)
            self.fair_data_path = Path(configs["out_path"]) / self.configs["name"] / "fair_data"
            os.makedirs(self.fair_data_path, exist_ok=True)
        else:
            self.model_path = None
            self.fair_data_path = None

    def fit(self, data):
        """
        Trains the pre-processing fairness intervention model on data.

        Parameters
        ----------
        data : pandas.DataFrame
            dataframe to be used for training;
            it needs to contain the sensitive information

        """
        pass

    def transform(self, data, y=None):
        """
        Using the pre-processing fairness model it generates the fair data representation for data.

        Parameters
        ----------
        data : pandas.DataFrame
           dataframe for which to generate the fair data representation;
           it does not need to contain the sensitive information

        Returns
        -------
        data_fair: pandas.DataFrame
          dataframe containing the fair transformed data representation as col_name + "_fair"
        """
        pass

    def fit_transform(self, data, y=None):
        """
        First trains the pre-processing fairness intervention model on the data if no model is already saved at out_path.
        Next, using the model it generates the fair data representation for data.

        Parameters
        ----------
        data : pandas.DataFrame
            dataframe to be used for training; if model is already saved no training will be performed
            if training needs to be performed, it needs to contain the sensitive information.

        Returns
        -------
        data_fair: pandas.DataFrame
           dataframe containing the fair transformed data representation as col_name + "_fair"

        """
        self.fit(data)
        return self.transform(data)

    def get_params(self, deep=True):
        """
        Ensure Scikit-Learn can access parameters.
        """
        return self.configs  # Return configs as parameters

    def set_params(self, **params):
        """
        Ensure Scikit-Learn can update parameters.
        """
        for key, value in params.items():
            self.configs[key] = value  # Update configs
        return self

    def is_start_train(self):
        """
        Returns True if the model is not saved.
        """
        if self.configs["out_path"] is None or (
                self.configs["out_path"] is not None and len(os.listdir(self.model_path)) == 0):
            start_train = True
        else:
            start_train = False
        return start_train

    def is_transform(self):
        """
        Returns True if the transformed data is not saved.
        """
        if self.configs["out_path"] is None:
            start_transform = True
            return start_transform
        else:
            if self.configs["file_name"] is not None:
                fair_data_file = os.path.join(self.fair_data_path, f'fair_{self.configs["file_name"]}_data.csv')
                return not os.path.exists(fair_data_file)
            else:
                return True




