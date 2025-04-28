from sklearn.base import BaseEstimator, TransformerMixin
from findhr.preprocess.metadata import JSONMetadata
from typing import List


class Mapping(BaseEstimator, TransformerMixin):
    """
    Base class for all mappings.

    Attributes
    ----------
    input_cols : list[str]
        list of input columns
    output_cols : list[str]
        list of output columns
    metadata_dict : dict[str:JsonColumnMetadata]
        metadata of the output columns
    """

    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None, metadata_dict=None):
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.metadata_dict = metadata_dict

    def fit(self, pair, input_cols: list[str] = None, output_cols: list[str] = None):
        """
        Fit the mapping to the data.

        Parameters
        ----------
        pair : tuple
            pair of data and metadata_dict
        input_cols : list[str]
            list of input columns
        output_cols : list[str]
            list of output columns

        Returns
        -------
        self : object
            Returns self.
        """
        pass

    def transform(self, X):
        """
        Transform the input data X according to the metadata_dict.

        Parameters
        ----------
        X : pandas.DataFrame
            input data to be transformed

        Returns
        -------
        X_new : dict[str:list]
            dictionary with the new columns as keys and the transformed data as values
        metadata_dict_new : dict[str:JsonColumnMetadata]
            metadata of the transformed data as a dictionary with element of res_col as keys
        """
        pass


class AttachMetadata(BaseEstimator, TransformerMixin):
    def __init__(self, metadata_dict: dict[str:JSONMetadata]):
        self.metadata_dict = metadata_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X, self.metadata_dict


class DetachMetadata(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.metadata_dict = None

    def fit(self, pair_X_meta, y=None):
        _, metadata_dict = pair_X_meta
        self.metadata_dict = metadata_dict
        return self

    def transform(self, pair_X_meta):
        return pair_X_meta[0]


class DerivedColumn(BaseEstimator, TransformerMixin):
    def __init__(self, mappings: dict[Mapping], verbose: bool = False):
        # dictionary of mappings
        self.mappings = mappings

        # list of output columns
        self.output_cols = None

        # metadata dictionary
        self.metadata_dict = None

        # verbose flag
        self.verbose = verbose
    def fit(self, pair_X_meta, y=None):
        X, metadata_dict = pair_X_meta
        self.output_cols = list(X.columns)
        self.metadata_dict = metadata_dict.copy()

        for (input_cols, output_cols), mapping in self.mappings.items():
            if self.verbose:
                print(f"Fitting mapping {mapping.__class__.__name__} for input columns {input_cols} and output columns {output_cols}")
            pair_input = (X[list(input_cols)], {col: metadata_dict[col] for col in input_cols})
            mapping.fit(pair_input, input_cols, output_cols)
            self.output_cols += list(mapping.output_cols)
            self.metadata_dict.update(mapping.metadata_dict)

        return self

    def transform(self, pair_X_meta):
        X, _ = pair_X_meta
        for (input_cols, output_cols), mapping in self.mappings.items():
            X_new = mapping.transform(X[list(input_cols)])
            X[list(output_cols)] = X_new[list(output_cols)].values
        return X, self.metadata_dict
