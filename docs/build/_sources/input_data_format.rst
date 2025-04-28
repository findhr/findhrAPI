.. _input_data_format:

Input Data Format
=================

Design Choices
--------------

There is no standard format nor univerally agreed format of the input data to an HR ranking system. 
For this reason, we make as few as possible assumptions on the input data, so that any organization can customize the APIs with its own data sources.
We assume that:

- Each raw data is provided by the user as a ``pandas.DataFrame``. Only a few columns are mandatory, while the number, the names and the types of all other columns are set by the user. Columns can have complex types, such as enumerated, arrays or compound types.

- Metadata about the format and the type/usage of columns in the raw data is provided by the user as a dictionary mapping column names to a metadata object.

The preparation of raw data and metadata is left to the user, as it is highly dependent on its information system sources.
We provide an example notebook (see :ref:`ex`) where raw data is loaded from a CSV file and metadata is defined directly in the notebook.
However, other solutions are possible, including loading data and metadata from a databases or from binary files, depending on the user choice.

Raw Data Sources
------------------
We assume three raw data sources formatted as ``pandas.DataFrame`` format and regarding candidates, jobs, and applications respectively.
The columns of the three *DataFrame*s describe the features of the candidates, of the jobs, and of the applications respectively.
Thus, in the following we use “column” and “feature” as interchangeable terms. 
Column values are Python objects satisfying the `JSON-schema <https://json-schema.org/>`_ format. 
Such a format allows for scalars (integers, numbers, strings) and compound objects (arrays/lists, dictionaries).

- **Candidates Data Source (CDS)**: A row for each candidate involved in the screening process. The mandatory column *candidate_id* uniquely identifies each candidate. Here it is an example:

.. image:: imgs/example_cds.png
  :width: 400
  :alt: Table with columns referring to the candidate data source

- **Jobs Data Source (JDS)**: A row for each job description to which a candidate can apply. The mandatory column *job_id* uniquely identifies each job description. Here it is an example:

.. image:: imgs/example_jds.png
  :width: 400
  :alt: Table with columns referring to the job description data source

- **Applications Data Source (ADS)**: A row for each application of a candidate to a job. The pair of mandatory columns *(candidate_id, job_id)* uniquely identifies the application of the candidate *candidate_id* to the job *job_id*. When the input is used for training a model, at least one mandatory column is required that specify the target feature to be learn by the model (see :ref:`target`). The name of the mandatory column is not fixed. Here it is an example with three target features:

.. image:: imgs/example_ads.png
  :width: 400
  :alt: Table with target and id columns referring to the application data source

Metadata
--------

For each raw data source, we assume that metadata is provided to specify the format, the type and the usage of each feature in the data source.
Metadata is in the form of a Python dictionary mapping the column name into an object of the class :class:`findhr.preprocess.metadata.JSONMetadata`.
Such a class has the following members to be provided in the constructor:

- **schema**: a dictionary following the `JSON-schema <https://json-schema.org/>`_ to describe the schema of the column values. The schema is used to test the conformance of the values to the expected format. The `JSON-schema <https://json-schema.org/>`_ is expressive enough to allow for scalars (integers, numbers, strings) and compound objects (arrays/lists, dictionaries), as well as for constraining their content (e.g., ranges for numbers, minimum length for arrays, optional and mandatory fields, and so on).


- **attr_type**: a string expressing the type of the feature as one of the following:

    - 'category': categorical feature type,

    - 'ordinal': ordinal feature type,

    - 'numeric': numerical feature type,

    - 'object': generic feature type.


- **attr_usage**: a string expressing the usage of the feature, as one of the following:

    - 'default': predictive feature for applicants, hard constraint for jobs,

    - 'optional': predictive feature for applicants, soft constraint for jobs,

    - 'protected': protected feature (w.r.t. fairness) of applicants,

    - 'sensitive': sensitive feature (w.r.t. privacy) of applicants,

    - 'target': target feature of the ranking model.


- **knowledge_base**: an optional member, with an unspecified data structure, used to encode domain knowledge useful for the computation of derived features, or for explanation purposes. For example, a URI to the `ESCO classification <https://esco.ec.europa.eu/en/about-esco/what-esco>`_.

The static function :class:`findhr.preprocess.metadata.validate_schema` validates the schemas of the metadata against the column of the raw data *DataFrame*s.


.. _target:

Target Feature and Ranking Type
-------------------------------

The Application Data Source (ADS) may include a feature labeled as ’target’ in the *attr_usage* metadata (the *target feature*).
Such a target feature can be used used for the training, validation, and testing of the ranking model using (historical) data on past selection procedures.
The *attr_type* metadata of such a feature depends on the type of ranking model as follows:

- 'category': the ranking model is a **binary probabilistic classifier** predicting a candidate’s probability of being shortlisted. The target feature assumes binary values with the meaning:

    - ‘0’: (not shortlisted) the candidate has not been admitted to the next phase of the recruiting process for the job ;

    - ‘1’: (shortlisted) the candidate has been admitted to the next phase of the recruiting process for the job.

- 'ordinal': the ranking model is a **ranker**, and the target feature assumes integer values corresponding to the ranking position of the candidate for the job (1 = best).

- 'numeric': the ranking model is a **scorer**, and the target features assume numeric values (typically in the range [0,1] or [0, 100]) meaning the score assigned to the candidate for the job (the higher the better).

In the following example of ADS there are three target features: *score*, which is of type numeric; *ranking*, which is of type ordinal; and *shortlisted*, which is of type category. They can be used for training a scorer, a ranker, or a binary classifier, respectively. Only one target feature is required for training a single model.

.. image:: imgs/example_ads.png
  :width: 400
  :alt: Table with target and id columns referring to the application data source


.. _ex:

Examples
--------

The notebook *Example_InputDataSources* in the :ref:`example_notebooks` shows an example of the three raw data sources and of their metadata.
The three raw data sources are joined into a single *DataFrame*.
Similarly, the respective three metadata are joined into a single dictionary.

Open Issues
-------------------------------------

We keep monitoring the generality of the assumed raw data and metadata formats.
In particular, the development of the demonstrators will be a useful testbed for this purpose.
