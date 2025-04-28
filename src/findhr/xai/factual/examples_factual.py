from findhr.xai.factual.factual import PostHocAgnosticExplainer
import shap

class PostHocAgnosticSHAPKernelExplainer(PostHocAgnosticExplainer):
    def __init__(self, predict_function, reference_data, explainer_class=shap.KernelExplainer):
        super().__init__(predict_function, reference_data, explainer_class)

    def fit(self, **kwargs):
        self.explainer = self.explainer_class(self.predict_function, self.reference_data, **kwargs)
        return self

    def _feature_importance(self, data, **kwargs):
        return self.explainer.shap_values(data, **kwargs)

    def _feature_importance_HU(self, data, metadata, **kwargs):
        return self.explainer.shap_values(data, **kwargs)