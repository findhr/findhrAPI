import numpy as np

from findhr.preprocess.metadata import JSONMetadata
from findhr.preprocess.mapping import Mapping
import pandas as pd
from collections import defaultdict
from typing import List
import itertools
import random


class IdentityMapping(Mapping):
    def fit(self, pair_X_meta, input_cols: List[str] = None, output_cols: List[str] = None):
        if len(input_cols) != 1 or len(output_cols) != 1:
            raise ValueError('incorrect number of columns')
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.metadata_dict = {self.output_cols[0]: JSONMetadata(
                                                    schema={'type': 'number', 'minimum': 0},
                                                    attr_type='numeric')}
        return self

    def transform(self, X):
        return pd.DataFrame.from_dict(
            {self.output_cols[0]: X[self.input_cols[0]]})

    def feature_importance(self):
        return {self.input_cols[0]: 1.}

class YearsOfExperience(Mapping):
    def fit(self, pair_X_meta, input_cols: List[str] = None, output_cols: List[str] = None):
        if len(input_cols) != 1 or len(output_cols) != 1:
            raise ValueError('incorrect number of columns')
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.metadata_dict = {self.output_cols[0]: JSONMetadata(
                                                    schema={'type': 'number', 'minimum': 0},
                                                    attr_type='numeric')}
        return self

    def transform(self, X):
        return pd.DataFrame.from_dict(
            {self.output_cols[0]: list(map(self.extract_sum_years_exp, X[self.input_cols[0]]))})

    @staticmethod
    def extract_sum_years_exp(exp_list):
        if len(exp_list) == 0:
            return 0
        return sum(float(exp['duration'].split(' ')[0]) for exp in exp_list)

    def feature_importance(self):
        return {self.input_cols[0]: 1.}

class YearsOfStudy(Mapping):

    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None, metadata_dict=None, kb_key='degree'):
        super().__init__(input_cols, output_cols, metadata_dict)
        self.kb_key = kb_key
        self.KB = None

    def fit(self, pair_X_meta, input_cols=None, output_cols=None):
        X, metadata = pair_X_meta

        if len(input_cols) != 1 or len(output_cols) != 1:
            raise ValueError('incorrect number of columns')
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.metadata_dict = {self.output_cols[0]: JSONMetadata(
                                                    schema={'type': 'number', 'minimum': 0},
                                                    attr_type='numeric')}

        # save the knowledge base
        self.KB = metadata[self.input_cols[0]].knowledge_base
        return self

    def transform(self, X):
        list_years = []
        for i, x in X.iterrows():
            list_years.append(
                self.extract_max_years_edu(x[self.input_cols[0]])
            )

        return pd.DataFrame.from_dict({self.output_cols[0]: list_years})

    def extract_max_years_edu(self, edu_list):
        if len(edu_list) == 0:
            return 0
        return max(self.KB[self.kb_key][edu[self.kb_key]] for edu in edu_list)

    def feature_importance(self):
        return {self.input_cols[0]: 1.}

class MinValueRequired(Mapping):
    """
    Example of a mapping that computes the minimum value required by a job offer.

    Attributes
    ----------
    input_cols: list[str]
        input_cols[0]: years of experience required by the job expressed as f'{years:float} years'
        input_cols[1]: years of experience of the candidate
    output_cols
    metadata_dict

    Methods
    -------
    fit(pair_X_meta, input_cols, output_cols)
        Fit the mapping to the data
    transform(X)
        Transform the input data X according to the metadata_dict
    feature_importance(self)
        Return the feature importance of the mapping
    """

    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None, metadata_dict=None):
        super().__init__(input_cols, output_cols, metadata_dict)

    def fit(self, pair_X_meta, input_cols=None, output_cols=None):
        if len(input_cols) != 2 or len(output_cols) != 1:
            raise ValueError('incorrect number of columns')
        self.input_cols = input_cols
        self.output_cols = output_cols

        # Note that we use integer because the json schema does not specify a boolean data type
        self.metadata_dict = {
            self.output_cols[0]: JSONMetadata(
                schema={'type': 'number'
                        # 'multipleOf': 1, 'minimum': 0, 'maximum': 1
                        },
                        attr_type='numeric')}
        return self

    def _check_reqs(self, job_exp_reqs, cv_exp):
        return int(cv_exp >= float(job_exp_reqs.split(' ')[0]))  # TBD: better way to compute duration

    def transform(self, X):
        return pd.DataFrame.from_dict(
            {self.output_cols[0]: map(self._check_reqs, X[self.input_cols[0]], X[self.input_cols[1]])})

    def feature_importance(self):
        return {self.input_cols[0]: 0.5, self.input_cols[1]: 0.5}


class RelevantExperienceForRole(Mapping):
    """
    Mapping that extracts the relevant experience of a candidate for a role indicated in the job requirements.

    Parameters:
    ----------
    input_cols: list[str]
        input_cols[0]: experience of the candidate
        input_cols[1]: list of job roles indicated as required by the job offer (each job role is a valid role)
    output_cols: list[str]
        output_cols[0]: relevant experience of the candidate for the job role
    metadata_dict: dict[str:JSONMetadata]
        metadata of the transformed data as a dictionary with element of res_col as keys
    exp_role_key: str
        key of the experience role in the experience list
    """
    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None, metadata_dict=None,
                 exp_role_key: 'str' = 'role'):

        super().__init__(input_cols, output_cols, metadata_dict)
        self.exp_role_key = exp_role_key
        self.KB = None

    def fit(self, pair_X_meta, input_cols=None, output_cols=None):
        X, metadata = pair_X_meta

        if len(input_cols) != 2 or len(output_cols) != 1:
            raise ValueError('incorrect number of columns')
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.metadata_dict = {self.output_cols[0]: metadata}

        return self

    def transform(self, X):
        list_exp = []
        for i, x in X.iterrows():
            list_exp.append(
                self.meaningful_job(x[self.input_cols[0]], x[self.input_cols[1]])
            )

        return pd.DataFrame.from_dict({self.output_cols[0]: list_exp})

    def meaningful_job(self, exp_list, job_exp_list):
        if len(exp_list) == 0 or len(job_exp_list) == 0:
            return []
        list_meaningful_exp = []

        for exp in exp_list:
            if exp[self.exp_role_key] in job_exp_list:
                list_meaningful_exp.append(exp)

        return list_meaningful_exp

    def feature_importance(self):
        """
        Return the feature importance of the mapping.
        In this case, the feature importance is
        - 1 for the first column (candidate data)
        - 0 for the second column (job data).
        """
        return {self.input_cols[0]: 0.5, self.input_cols[1]: 0.5}
class ExtractListOfProperty(Mapping):
    """
    Mapping that extracts the properties of a list of dictionary corresponding to a specific key
    Example:
         input data
             [
                 {'institution': 'Complutense University Of Madrid',
                  'start_date': 'January 2022', 'end_date': 'January 2023',
                  'degree': 'Bachelor Degree In Political Sciences'},
                 {'institution': 'Complutense University Of Madrid',
                  'start_date': 'January 2023', 'end_date': 'January 2024',
                  'degree': 'Master's Degree In Political Sciences'}
    ]

    output data
        [Bachelor Degree In Political Sciences, Master's Degree In Political Sciences]

    Parameters:
    ----------
    input_cols: list[str]
        input_cols[0]: column assumed as list[dict]
    output_cols: list[str]
        output_cols[0]: list of properties of the input_cols[0] list
    metadata_dict: dict[str:JSONMetadata]
        metadata of the transformed data as a dictionary with element of res_col as keys
    property: str
        key of the dict element to extract
    """

    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None, metadata_dict=None,
                 property_key: 'str' = 'degree'):

        super().__init__(input_cols, output_cols, metadata_dict)
        self.property_key = property_key
        self.KB = None

    def fit(self, pair_X_meta, input_cols=None, output_cols=None):
        if len(input_cols) != 1 or len(output_cols) != 1:
            raise ValueError('incorrect number of columns')
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.metadata_dict = {self.output_cols[0]:
            JSONMetadata(
                schema={'type': 'array',
                        'items': {'type': 'string'}},
                attr_type='object',
                attr_usage='default')
            }

        return self

    def transform(self, X):
        list_exp = []
        for i, x in X.iterrows():
            list_exp.append(
                self.extract_property_list(x[self.input_cols[0]])
            )

        return pd.DataFrame.from_dict({self.output_cols[0]: list_exp})

    def extract_property_list(self, list_dict):
        if len(list_dict) == 0:
            return []

        property_list = []
        for el in list_dict:
            property_list.append(el[self.property_key])

        # remove duplicates
        return list(set(property_list))

    def feature_importance(self):
        """
        Return the feature importance of the mapping.
        """
        return {self.input_cols[0]: 1}


class ExtractMonthDurationJob(Mapping):

    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None, metadata_dict=None,
                 duration_key='duration_months'):
        super().__init__(input_cols, output_cols, metadata_dict)
        self.duration_key = duration_key
        self.KB = None

    def fit(self, pair_X_meta, input_cols=None, output_cols=None):
        if len(input_cols) != 1 or len(output_cols) != 1:
            raise ValueError('incorrect number of columns')
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.metadata_dict = {self.output_cols[0]: JSONMetadata(
                                                    schema={'type': 'number', 'minimum': 0},
                                                    attr_type='numeric')}

        return self

    def transform(self, X):
        list_years = []
        for i, x in X.iterrows():
            list_years.append(
                self.extract_months_job(x[self.input_cols[0]])
            )

        return pd.DataFrame.from_dict({self.output_cols[0]: list_years})

    def extract_months_job(self, exp_list):
        if len(exp_list) == 0:
            return 0
        duration = 0

        for exp in exp_list:
            duration += exp[self.duration_key]

        return duration

    def feature_importance(self):
        """
        Return the feature importance of the mapping.
        In this case, the feature importance is
        - 1 for the first column (candidate data)
        - 0 for the second column (job data).
        """
        return {self.input_cols[0]: 1.0}

class MatchMapping(Mapping):
    """
    Mapping that matches the job_offer data with the curriculum data.
    It checks if the curriculum feature (cv_data) satisfies the requirement of the job offer (job_req).
    """
    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None, metadata_dict=None, keyword_any: str = 'Any'):
        super().__init__(input_cols, output_cols, metadata_dict)
        self.keyword_any = keyword_any
        self.KB = None
        self.schema_input_cols = None

    def fit(self, pair_X_meta, input_cols=None, output_cols=None):
        X, input_metadata = pair_X_meta

        if len(input_cols) != 2 or len(output_cols) != 1:
            raise ValueError(f'incorrect number of columns:\n'
                             f'- input_cols should be 2 (actual list: {input_cols});'
                             f'- output_cols should be 1 (actual list: {output_cols})')
        self.input_cols = input_cols
        self.output_cols = output_cols
        # Note that we use integer because the json schema does not specify a boolean data type
        self.metadata_dict = {
            self.output_cols[0]: JSONMetadata(
                schema={'type': 'number'
                        },
                        attr_type='numeric')}

        self.schema_input_cols = {col: input_metadata[col].schema for col in input_cols}

        self.KB = input_metadata[self.input_cols[0]].knowledge_base
        return self

    def _check_reqs(self, job_req, cv_data):
        return NotImplementedError()

    def transform(self, X):
        return pd.DataFrame.from_dict(
            {self.output_cols[0]: map(self._check_reqs, X[self.input_cols[0]], X[self.input_cols[1]])})

    def compute_cf(self, job_req, cv_data, fitness_output):
        """
        Compute the counterfactual of the cv_data to satisfy the job_req.
        """
        pass



class MatchBinary(MatchMapping):
    """
    Mapping that matches the job_offer data with the curriculum data.
    It checks if the curriculum feature (cv_data) satisfies the requirement of the job offer (job_req).

    In particular, it assumes that the feature is binary.
    The mapping checks if the cv_data feature is equal to the job_offer requirement.
    - If job_req or the cv_data is the keyword for the requirement "any" (i.e., all accepted), then it returns 1.
    - Else,
    - - if knowledge base is not provided, it checks using directly the passed values
    - - otherwise, it uses the knowledge base to check the values.
    """
    def _check_reqs(self, job_req, cv_data):
        if job_req == self.keyword_any or cv_data == self.keyword_any:
            return 1
        else:
            if self.KB is None:
                return int(cv_data == job_req)
            else:
                return int(self.KB[cv_data] == self.KB[job_req])

    def compute_cf(self, job_req, cv_data, fitness_output):
        """
        Compute the counterfactual of the cv_data to satisfy the job_req.
        e.g., if cv_data = 0 and job_req = 1, then the counterfactual is 1.
        """
        if fitness_output == 1:
            # If the fitness output is 1, then the counterfactual must be equal to job_req
            return job_req
        elif fitness_output == 0:
            # If the fitness output is 0, then the cv_data must be different from job_req
            if job_req == self.keyword_any:
                raise ValueError('Cannot compute counterfactual for job requirement "Any" keyword and '
                                  'fitness_output == 0')
            else:
                if self.KB is not None:
                    # Find another value that is not equal to the job requirement
                    return random.choice([v for k, v in self.KB.items() if v != self.KB[job_req]])
                    # for k, v in self.KB.items():
                    #     if v != self.KB[job_req]:
                    #        list_values.append(k)
                    # return random.choice(list_values)
                elif self.schema_input_cols is not None:
                    if 'enum' not in self.schema_input_cols[self.input_cols[0]].keys():
                        raise ValueError('Cannot compute counterfactual for fitness_output == 0 without knowledge base '
                                         'or enum in JSONMetadata for input column 0')
                    else:
                        possible_values = self.schema_input_cols[self.input_cols[0]]['enum']
                        return random.choice([v for v in possible_values if v != job_req])
                else:
                    raise ValueError('Cannot compute counterfactual for fitness_output == 0 without knowledge base '
                                     'or schema_input_cols')

        else:
            raise ValueError('Fitness output must be 0 or 1 for a MatchBinary mapping')


class MatchOrdinal(MatchMapping):
    """
    Mapping that matches the job_offer data with the curriculum data.
    It checks if the curriculum feature (cv_data) satisfies the requirement of the job offer (job_req).

    In particular, it assumes that the feature is ordinal.
    The mapping checks if the cv_data feature is greater or equal than the same requirement in job_req.
    - If the job_req is the keyword for the requirement "any" (i.e., every cv is accepted), then it returns 1.
    - Else,
    - - if knowledge base is not provided, it checks using directly the passed values
    - - otherwise, it uses the knowledge base to check the values.
    """
    def _check_reqs(self, job_req, cv_data):
        if job_req == self.keyword_any:
            return 1
        else:
            if self.KB is None:
                return int(cv_data >= job_req)
            else:
                return int(self.KB[cv_data] >= self.KB[job_req])

    def compute_cf(self, job_req, cv_data, fitness_output):
        """
        Compute the counterfactual of the cv_data to satisfy the job_req.
        e.g., if cv_data = 2 and job_req = 3, meaning >=3, then the counterfactual is 3.
        """
        if fitness_output == 1:
            # If the fitness output is 1, then the counterfactual must be at least equal to the job_req
            if self._check_reqs(job_req, cv_data):
                # If the cv_data already satisfies the job requirement, then return cv_data
                return cv_data
            else:
                # else return the job_req, which is the minimum value required and closest to cv_data,
                # assuming an increasing order
                return job_req
        elif fitness_output == 0:
            # If the fitness output is 0, then the counterfactual must be different from job_req
            if job_req == self.keyword_any:
                raise ValueError('Cannot compute counterfactual for job requirement "Any" keyword '
                                  'and fitness_output == 0')
            else:
                if self.KB is None:
                    raise ValueError('Cannot compute counterfactual for fitness_output == 0 without knowledge base')
                else:
                    # Find the key corresponding to the max value in the KB less than job_req
                    list_pairs = [(k, v) for k, v in self.KB.items() if v < self.KB[job_req]]
                    sorted(list_pairs, key=lambda x: x[1])
                    return list_pairs[-1][0]
                    # for k, v in self.KB.items():
                    #     if v != self.KB[job_req]:
                    #        list_values.append(k)
                    # return random.choice(list_values)
        else:
            raise ValueError('Fitness output must be 0 or 1 for a MatchOrdinal mapping')


class MatchFeatureInclusion(MatchMapping):
    """
    Mapping that matches the job_offer data with the curriculum data.
    It checks if the curriculum feature (cv_data) satisfies the requirement of the job offer (job_req).

    In particular, it assumes that the job_offer expresses a requirement as a numeric interval [min_value, max_value],
    whereas the cv expresses a numerical single value
    The mapping checks if the cv_data feature is inside the numeric interval of the job_offer.
    - If the job_req is the keyword for the requirement "any" (i.e., every cv is accepted), then it returns 1.
    - Else,
    - - if the cv_data satisfies the job requirement, it returns 1
    - - otherwise, it returns 0
    """
    def _check_reqs(self, job_req, cv_data, kb=None):
        if job_req == 'Any':
            return 1
        elif job_req[0] <= cv_data <= job_req[1]:
            return 1
        else:
            return 0

class MatchFeatureAtLeastInList(MatchMapping):
    """
    Mapping that matches the job_offer data with the curriculum data.
    It checks if the curriculum feature (cv_data) satisfies the requirement of the job offer (job_req).

    In particular, it assumes that the job_offer expresses a requirement as a list of strings ["s1","s2",...];
    whereas the cv expresses a list of strings ["s1","s2",...] as well.
    The mapping checks if the at least one value of the cv_data  is present in the job_offer requirement.
    - If the job_req is an empty list [] (i.e., every cv is accepted), then it returns 1.
    - else,
    - - if at least one cv_data satisfies the job requirement, it returns 1
    - - otherwise, it returns 0
    """
    def _check_reqs(self, job_req, cv_data, kb=None):
        if len(job_req) == 0:
            return 1
        else:
            for val in cv_data:
                if val in job_req:
                    return 1
            return 0

class MatchFeatureInclusionSeparated(Mapping):
    """
    Mapping that matches the job_offer data with the curriculum data.
    It checks if the curriculum feature (cv_data) satisfies the requirement of the job offer (job_req_min, job_offer_max).

    In particular, it assumes that the job_offer expresses a requirement as a numeric interval [min_value, max_value],
    whereas the cv expresses a numerical single value
    The mapping checks if the cv_data feature is inside the numeric interval of the job_offer.
    - If the job_req is the keyword for the requirement "any" (i.e., every cv is accepted), then it returns 1.
    - Else,
    - - if the cv_data satisfies the job requirement, it returns 1
    - - otherwise, it returns 0
    """
    """
        Mapping that matches the job_offer data with the curriculum data.
        It checks if the curriculum feature (cv_data) satisfies the requirement of the job offer (job_req).
        """

    def __init__(self, input_cols: List[str] = None, output_cols: List[str] = None, metadata_dict=None,
                 keyword_any: str = 'Any', min_delta_value: int = 1, ):
        super().__init__(input_cols, output_cols, metadata_dict)
        self.keyword_any = keyword_any
        self.KB = None
        self.schema_input_cols = None
        self.min_delta_value = min_delta_value
        assert self.min_delta_value > 0, 'min_delta_value must be greater than 0'

    def fit(self, pair_X_meta, input_cols=None, output_cols=None):
        X, input_metadata = pair_X_meta

        if len(input_cols) != 3 or len(output_cols) != 1:
            raise ValueError(f'incorrect number of columns:\n'
                             f'- input_cols should be 3 (actual list: {input_cols});'
                             f'- output_cols should be 1 (actual list: {output_cols})')
        self.input_cols = input_cols
        self.output_cols = output_cols
        # Note that we use integer because the json schema does not specify a boolean data type
        self.metadata_dict = {
            self.output_cols[0]: JSONMetadata(
                schema={'type': 'number'
                        },
                attr_type='numeric')}
        self.schema_input_cols = {col: input_metadata[col].schema for col in input_cols}
        self.KB = input_metadata[self.input_cols[0]].knowledge_base
        return self

    def _check_reqs(self, job_req_min, job_req_max, cv_data, kb=None):
        if job_req_min == 'Any' or job_req_max == 'Any':
            return 1
        elif job_req_min <= cv_data <= job_req_max:
            return 1
        else:
            return 0

    def transform(self, X):
        return pd.DataFrame.from_dict(
            {self.output_cols[0]: map(self._check_reqs,
                                      X[self.input_cols[0]], X[self.input_cols[1]], X[self.input_cols[2]]
                                      )})

    def compute_cf(self, job_req_min, job_req_max, cv_data, fitness_output):
        """
        Compute the counterfactual of the cv_data to satisfy the job_req.
        """
        if fitness_output == 1:
            # If fitness output == 1, then the counterfactual must be inside the bounds of (job_req_min, job_req_max)
            if self._check_reqs(job_req_min, job_req_max, cv_data):
                # If the cv_data already satisfies the job requirement, then return cv_data
                return cv_data
            else:
                # else return the closer value to cv_data inside job_req bounds
                return job_req_min if abs(cv_data - job_req_min) < abs(cv_data - job_req_max) else job_req_max
        elif fitness_output == 0:
            # If fitness output == 0, then the counterfactual must be outside the bounds of (job_req_min, job_req_max)
            if not self._check_reqs(job_req_min, job_req_max, cv_data):
                # If the cv_data does not satisfy the job_req then return cv_data
                return cv_data
            elif job_req_min == self.keyword_any or job_req_max == self.keyword_any:
                raise ValueError('Cannot compute counterfactual for job requirement "Any" keyword and '
                                  'fitness_output == 0')
            else:
                # Find the value outside the bounds of (job_req_min, job_req_max) closer to cv_data
                return job_req_min - self.min_delta_value if abs(cv_data - job_req_min) < abs(cv_data - job_req_max) \
                    else job_req_max + self.min_delta_value


class MatchFeatureSet(MatchMapping):
    """
    Mapping that matches the job_offer data with the curriculum data.
    It checks if the curriculum feature (cv_data) satisfies the requirement of the job offer (job_req).

    In particular, it assumes that the job_req expresses a requirement as a list of strings ["s1","s2",...];
    similarly the cv expresses a feature as a list of strings.
    The mapping checks if the proportion of cv_data features that are inside the job_req list.
    Note that the empty list assigned to job_req defines the absence of the requirement.
    """
    def _check_reqs(self, job_req, cv_data, kb=None):
        if len(job_req) == 0:
            return 0
        return len(set(cv_data) & set(job_req)) / len(job_req)

    def compute_cf(self, job_req, cv_data, fitness_output):
        """
        Compute the counterfactual of the cv_data to satisfy the job_req.
        """
        if fitness_output < 0 or fitness_output > 1:
            raise ValueError('Fitness output must be in [0, 1] for a MatchFeatureSet mapping')
        elif fitness_output == 1:
            # If fitness output == 1, then the counterfactual must contain the values in the job_req set
            if self._check_reqs(job_req, cv_data) == 1:
                # If the cv_data already satisfies the job requirement, then return cv_data
                return cv_data
            else:
                # else return the union of cv_data and job_req
                return list(set(cv_data) | set(job_req))

        else:
            delta_fitness = fitness_output - self._check_reqs(job_req, cv_data)
            # print(f'delta_fitness = {delta_fitness}')
            if delta_fitness == 0:
                return cv_data
            elif delta_fitness > 0:
                # if the proportion of cv_data features that are inside the job_req list is less than fitness_output,
                # then we need to add the missing values from job_req to cv_data until the proportion is satisfied

                output_data = cv_data.copy()
                num_elements_to_add = int(delta_fitness * len(job_req) + 0.5)
                # print('delta_fitness = ', delta_fitness)
                elements_to_add = random.choices(list(set(job_req) - set(cv_data)), k=num_elements_to_add)
                # print(f'output_data = {output_data}')
                # print(f'num_elements_to_add = {num_elements_to_add}')
                # print(f'elements_to_add = {elements_to_add}')

                output_data.extend(elements_to_add)

                return output_data
            else:
                # if the proportion of cv_data features that are inside the job_req list is greater than fitness_output,
                # then we need to remove the exceeding values from cv_data until the proportion is satisfied
                output_data = cv_data.copy()
                num_elements_to_remove = int(-delta_fitness * len(cv_data) + 0.5)
                elements_to_remove = random.choices(cv_data, k=num_elements_to_remove)
                # print(f'delta_fitness = {delta_fitness}')
                # print(f'output_data = {output_data}')
                # print(f'num_elements_to_remove = {num_elements_to_remove}')
                # print(f'elements_to_remove = {elements_to_remove}')

                for el in elements_to_remove:
                    output_data.remove(el)
                return output_data





