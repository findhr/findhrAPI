from findhr.xai.counterfactual.dice_ml.model_interfaces.base_model import BaseModel
from findhr.xai.counterfactual.dice_ml.constants import ModelTypes
import numpy as np
from findhr.xai.counterfactual.dice_ml.utils.exception import SystemException


class LGBMRankerPipelineModel(BaseModel):

    def __init__(self, model=None,
                 model_path='',
                 backend={'explainer': 'dice_xgboost.DiceGenetic', 'model': "lgbmranker_pipeline_model.LGBMRankerPipelineModel"},
                 func=None,
                 kw_args=None):
        super().__init__(model=model, model_path=model_path, backend=backend, func=func, kw_args=kw_args)
        self.top_k = kw_args['top_k']
        # self.features_dtype = kw_args['features_dtype']

    def predict(self, X):
        # Enforce feature dtypes before prediction
        # for col, dtype in self.features_dtype.items():
        #     X[col] = X[col].astype(dtype)
        return self.model.predict(X)

    def predict_proba(self, X):
        # TODO: solve the issue with datatypes
        # print(X.info())
        # for col in X.columns:
        #      X[col] = X[col].astype(float)

        # We do not have probabilities, but we can return if the candidate is in the top k
        return self.model.predict(X) <= self.top_k

    def get_output(self, input_instance, model_score=True):
        """returns prediction probabilities for a classifier and the predicted output for a regressor.

        :returns: an array of output scores for a classifier, and a singleton
        array of predicted value for a regressor.
        """
        input_instance = self.transformer.transform(input_instance)
        if model_score:
            if self.model_type == ModelTypes.Classifier:
                return self.predict(input_instance) <= self.top_k
            else:
                return self.predict(input_instance)
        else:
            return self.predict(input_instance)
        # if model_score:
        #     if self.model_type == ModelTypes.Classifier:
        #         return np.atleast_2d(self.model.predict(input_instance) <= self.top_k)
        #     else:
        #         return np.atleast_2d(self.model.predict(input_instance))
        # else:
        #     return np.atleast_2d(self.model.predict(input_instance))

    def get_gradient(self):
        raise NotImplementedError

    def get_num_output_nodes(self, inp_size):
        return 1

    def get_num_output_nodes2(self, input_instance):
        return 1

