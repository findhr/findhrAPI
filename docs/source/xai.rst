eXplainable AI (XAI)
====================

Design Choices
--------------

Explanations aim to provide understandable information about the ranking produced by the HR ranking system. An explanation can be factual (*“Why the decision?”*) or counterfactual (*“Why not another decision?”*). In addition, the explanations should take into account the specific stakeholder's information needs (expectations, assumed knowledge, level of detail, etc.).

To clarify the explanation process, it is useful to identify three data domains.

- The *Human-Understandable Data Domain (HUDD)* consists of the domain of raw data sources (see :ref:`input_data_format`). This domain is the interface to human stakeholders, who are assumed to clearly understand the structure, semantics, and connections among features.

- The *Feature Engineering Data Domain (FEDD)* is the domain of the output of the preprocessing phase (see :ref:`feature_preprocessing`). It includes features that are derived from the HUDD by design choices of the developers of the HR system. Such features are fed in input to the ranking model.

- The *Ranking Output Data Domain (RODD)* is the domain of the output of the ranking model. For scoring models and probabilistic classifiers models, it consists of score values to be sorted for producing a ranking for each job position. For ranking models, it consists of ranks starting from 1 (the best) to the number N (the worst) candidates to a each job position.

Consequently, each explanation function will be specific to the data domain over which the explanation has to be provided, e.g., FEDD for developers and HUDD for candidates and recruiters. Moreover, explanations may operate differently based on the stakeholder they are addressed to; for example, feature importance for candidates should be computed by keeping the job description features constant.

XAI Architecture
-----------------------

HR ranking systems are the result of a pipeline consisting of a preprocessing phase and a model training phase producing a ranking model (a scorer, a probabilistic classifier, or a ranking model).
Altogether, this is the explicand to be explained.
We distinguish three architectures of the HR ranking systems, and their respective explanation design:

- **Black-box**: both the preprocessing and the ranking model provide no information about their inner working rationale.
  We do not have access to the metadata information of the preprocessing; furthermore, the ranking model is not interpretable, yet it can be accessed (e.g., a neural network can be accessed in its components, but it is not interpretable).
  The explanation method can only access the predict function, which passed through both preprocessing and model prediction. The explanation is then made ex-post to map back directly from the RODD to HUDD.
  In summary, the explanation describes each ranking outcome in the human-understandable domain with no knowledge of preprocessing and ranking.

- **White-box**: the pipeline allows backtracking of the data and metadata flow at each step.
  The ranking model is interpretable (either by-design or as an integrated interpretation method), i.e., it provides a clearly understandable interpretation of itself.
  The explanation method can compose the explanation of the ranking model with the metadata of the preprocessing steps to obtain an interpretation of each prediction from the RODD to the HUDD.

- **Gray-box**: the preprocessing pipeline is as in the white-box category, whereas the ranking model is as in the black-box category.
  In this case, a post-hoc model-agnostic explanation is applied to the ranking model to obtain an explanation in the FEDD.
  The final HUDD explanation is obtained by composing the FEDD explanation with the inverse tracking method exposed by the pipeline of the preprocessing phase, which accounts for the metadata flow.

Explanations
--------------

We distinguish:

- **Factual explanation**: We assign a numeric value (feature importance) to each feature of an instance x based on the following cases:

  - if :math:`\mathbf{x}` is from HUDD and the explicand is a black-box, then the prediction is in RODD. Therefore the numeric value represents the feature importance with respect to the ranking.

  - if :math:`\mathbf{x}` is from HUDD and the explicand is a gray-box, then the preprocessing yields an instance in the FEDD. In this case, the numeric value represents the importance of the features in :math:`\mathbf{x}` with respect to the preprocessed features.

  - if :math:`\mathbf{x}` is from FEDD and the explicand is a gray-box, then the prediction is in the RODD. Here, the numeric value represents the importance of the preprocessed features with respect to the ranking.

  The last two cases are not mutually exclusive, i.e., there can be an explanation for an instance of HUDD mapping feature importance towards preprocessed instance in FEDD, and another explanation mapping feature importance of the preprocessed features with regard to the ranking in RODD.

  We provide base classes :class:`findhr.xai.factual.PostHocAgnosticExplainer` and :class:`findhr.xai.factual.PostHocSpecificExplainer` for model-agnostic and model-specific explainers respectively. A wrapper around the `KernelExplainer` of the `shap` package is provided in :class:`findhr.xai.examples_factual.PostHocAgnosticSHAPKernelExplainer`. Wrapping allows us to reuse the existing explanation methods from the literature. Specialized classes for factual explanation will be developed in the FINDHR project.

- **Counterfactual explanation**: Consider an instance :math:`\mathbf{x}` belonging to the HUDD or to the FEDD. Once in input to the explicand, produces an output (a score, a probability, or a rank position). A counterfactual explanation consists of one (or more) other instance(s) :math:`\mathbf{x}_c` which highlights the features to be modified to obtain a different output. Modification of features should be minimal (so that :math:`\mathbf{x}` and :math:`\mathbf{x}_c` are close each other) and actionable (i.e., feasible). In particular, counterfactual explanations help understand what a candidate could to change to improve the suitability for the job by setting the different output as follows:

    - for a scorer (where a prediction is a score), an increased score for :math:`\mathbf{x}_c`;
    
    - for a binary probabilistic classifier (where a prediction is the probability of being shortlisted), an increased probability for :math:`\mathbf{x}_c`;
    
    - for a ranking model (where a prediction is the rank of the candidate to the job), an increased rank (e.g., to be in the top-k positions) for :math:`\mathbf{x}_c`.

A further input for the explainability counterfactual process is the cost to modify certain features, which maybe different from one candidate to another.

Examples
--------

The notebook Example_FactualExplanation in the :ref:`example_notebooks` shows an example of factual explanation built on top of the preprocessing and scorer (regression) model from the notebook Example_Preprocessing.

Open Issues
-----------

1. We want to allow factual explanations expressed by feature importance to be implemented by defining inverse transformations of the mappings of the preprocessing phase.
   Clearly, not all the mappings are invertible, therefore an idea is to design a method to standardize the computation of the feature importance also through non-invertible mappings.

2. The implementation of an interpretable-by-design ranking model accounting for fairness is not straightforward.
   The main problem to be addressed is how to design a method that is flexible enough to account for in-processing fairness and it is interpretable-by-design.
