import shap
from sklearn.base import BaseEstimator

class PostHocAgnosticExplainer(BaseEstimator):
    """
    Base class for post-hoc model agnostic explainers.

    Attributes
    ----------
    predict_function : function
        The function that predicts the output of the model.
    reference_data : pandas.DataFrame
        The reference data used to explain the model.
    explainer_class : class
        The explainer class to use.
    explainer : object
        The instantiated explainer object to be filled by calling the fit method

    Methods
    -------
    fit()
        Fit the explainer object.
    predict(data, metadata=None)
        Predict the feature importance of the data, with or without metadata.

    """
    def __init__(self, predict_function, reference_data, explainer_class):
        self.predict_function = predict_function
        self.reference_data = reference_data
        self.explainer_class = explainer_class
        self.explainer = None

    def fit(self):
        """
        Interface method to fit the explainer object.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        return self

    def predict(self, data, metadata=None, **kwargs):
        """
        Interface method to predict the feature importance of the data, with or without metadata.
        """
        if metadata is None:
            return self._feature_importance(data, **kwargs)
        else:
            return self._feature_importance_HU(data, metadata, **kwargs)

    def _feature_importance(self, data, **kwargs):
        raise NotImplementedError()

    def _feature_importance_HU(self, data, metadata, **kwargs):
        raise NotImplementedError()


class PostHocSpecificExplainer(BaseEstimator):
    def __init__(self, model, reference_data, explainer_class):
        self.model = model
        self.reference_data = reference_data
        self.explainer_class = explainer_class
        self.explainer = None

    def fit(self, kwargs=None):
        return self

    def predict(self, data, metadata=None, **kwargs):
        """
        Interface method to predict the feature importance of the data, with or without metadata.
        """
        if metadata is None:
            return self._feature_importance(data, **kwargs)
        else:
            return self._feature_importance_HU(data, metadata, **kwargs)

    def _feature_importance(self, data, **kwargs):
        raise NotImplementedError()

    def _feature_importance_HU(self, data, metadata, **kwargs):
        raise NotImplementedError()

