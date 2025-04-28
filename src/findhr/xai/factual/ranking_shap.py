# Inspired from https://github.com/MariaHeuss/RankingShap/blob/main/approaches/ranking_shap.py
import numpy as np
import shap
from scipy.stats import kendalltau
import pandas as pd
from functools import partial
from shap.utils._legacy import convert_to_model
from findhr.xai.utils import rank_list



# Our implementation of KernelShap needs a re-initialization of the model.predict function
# for each query that it is explaining. Hence for the initialization we will use a place-holder
# function that returns an error if not re-initialized with another function.
def placeholder_predict(array):
    Warning(
        "The model.predict function needs to be defined for each query individually."
    )
    return np.array([0] * len(array))


def new_model_predict_val(
    array,
    original_model_predict,
    query_features,
    similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    mixed_type_input=False,
):
    # Determine ranking for current query
    pred = original_model_predict(query_features)
    og_rank = rank_list(pred)

    # Adjust feature vectors: substitute not None features for all candidate documents with background/0 value
    if not mixed_type_input:
        adjusted_features = np.array(
            [[np.where(pd.isna(a), doc, a) for doc in query_features] for a in array]
        )
    else:
        adjusted_features = np.array(
            [
                [pd.Series(doc).where(pd.isna(a), a).values for doc in query_features]
                for a in array
            ]
        )  # Need to use pd for mixed type array.

    scores = []
    for features_background_sample in adjusted_features:
        # Determine ranking for adjusted document feature vectors
        new_pred = original_model_predict(features_background_sample)
        new_rank = rank_list(new_pred)
        scores.append(similarity_coefficient(og_rank, new_rank))
    return np.array(scores)


class RankingShap:
    """
    This class implements the RankingShap method for feature attribution in ranking problems.
    It uses the SHAP library to compute Shapley values for feature importance.

    Parameters:
    ----------
    permutation_sampler : str
        The type of permutation sampler to use. Options are "kernel" or "sampling".
    background_data : np.ndarray
        The background data used for computing Shapley values.
    original_model : callable
        The original model used for predictions.
    explanation_size : int, optional
        The number of features to include in the explanation. Default is 3.
    name : str, optional
        The name of the explainer. Default is an empty string.
    nsample_permutations : int or str, optional
        The number of permutations to sample. Default is "auto".
    rank_similarity_coefficient : callable, optional
        The function used to compute the rank similarity coefficient. Default is Kendall's tau.
    """
    def __init__(
        self,
        permutation_sampler,
        background_data,
        original_model,
        explanation_size=3,
        name="",
        nsample_permutations="auto",
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    ):
        assert permutation_sampler in ["kernel", "sampling"]
        self.permutation_sampler = permutation_sampler
        self.background_data = background_data
        self.original_model = original_model
        # self.model = partial(, ),
        self.explanation_size = explanation_size

        self.nsamples = nsample_permutations
        self.name = name

        self.feature_shape = np.shape(background_data[0])
        self.num_features = len(background_data[0])

        self.explainer = self.get_explainer()
        self.rank_similarity_coefficient = rank_similarity_coefficient
        self.new_model_predict = partial(
            new_model_predict_val,
            original_model_predict=original_model,
            similarity_coefficient=rank_similarity_coefficient,
        )
        self.feature_attribution_explanation = None
        self.feature_selection_explanation = None

    def get_explainer(self):
        if self.permutation_sampler == "kernel":
            shap_explainer = shap.KernelExplainer(
                placeholder_predict, self.background_data, nsamples=self.nsamples,
                explanation_size=self.explanation_size
            )
        elif self.permutation_sampler == "sampling":
            shap_explainer = shap.SamplingExplainer(
                placeholder_predict, self.background_data, nsamples=self.nsamples
            )
        else:
            raise ValueError("Permutation sampler not recognized.")
        return shap_explainer

    def get_query_explanation(self, query_features, query_id=""):
        """
        Get the explanation for a specific query.

        Parameters:
        ----------
        query_features : np.ndarray
            The feature vector for the query.
        query_id : str, optional
            The ID of the query. Default is an empty string.

        Returns:
        -------
        dict
            A dictionary containing the feature importance scores.
        """

        # Set the query dependent model predict function
        self.explainer.model = convert_to_model(
            partial(self.new_model_predict, query_features=query_features)
        )

        # This model predict takes over part of the masking and substitutes the not None,
        # corresponding features of all documents with values from the feature original feature vector
        # and rerank wrt that newly obtained vector.
        vector_of_nones = np.array([np.full(self.feature_shape, None)])

        if self.permutation_sampler == "kernel":
            exp = self.explainer.shap_values(vector_of_nones, nsamples=self.nsamples)[0]
        else:
            exp = self.explainer(vector_of_nones, nsamples=self.nsamples).values[0]

        exp_dict = {
            i + 1: exp[i] for i in range(len(exp))
        }  # We start numbering the features with 1.

        exp_dict = sorted(exp_dict.items(), key=lambda item: item[1], reverse=True)
        return exp_dict