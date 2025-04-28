import jsonschema
import numpy as np


class JSONMetadata:
    """
    Store the attribute metadata of CVs or job descriptions in JSON format.

    Attributes
    ----------
    schema : dict
        JSon schema of the attribute according to https://json-schema.org/
    attr_usage : str
        description of the usage of the attribute. Possible values:
            'default' (default, predictive attribute, hard constraint),
            'optional' (predictive attribute, soft constraint)
            'protected' (fairness, protected attribute),
            'sensitive' (privacy, sensitive attribute),
            'target' (predictive attribute, target attribute)
    attr_type : str, optional
        description of the type of the attribute. Possible values:
            'category' (categorical attribute, target=classification),
            'ordinal' (ordered attribute, target=ranking),
            'numeric' (numerical attribute, target=scoring),
            'object' (structured attribute, not target),
    knowledge_base : str, optional
        additional information about the attribute to be used in the pipeline for processing and deriving fitness values
    """
    def __init__(self, schema: dict,
                 attr_usage: str = 'default',
                 attr_type: str = 'object',
                 knowledge_base=None):

        # Defines the schema of the column according to the convention of https://json-schema.org/
        self.schema = schema

        # Description of the column type.
        if attr_type not in {'category', 'ordinal', 'numeric', 'object'}:
            raise TypeError('attr_type parameter unknown: '+attr_type)
        self.attr_type = attr_type

        # Description of the column usage.
        if attr_usage not in {'default', 'optional', 'protected', 'sensitive', 'target'}:
            raise TypeError('attr_usage parameter unknown: '+attr_usage)
        self.attr_usage = attr_usage

        # Additional information about the column to be used in the pipeline for processing and deriving fitness values
        self.knowledge_base = knowledge_base

    def __repr__(self):
        return f'\n\tSCHEMA = {self.schema}\n' \
               f'\tATTR_TYPE = {self.attr_type}\n' \
               f'\tATTR_USAGE = {self.attr_usage}\n' \
               f'\tKNOWLEDGE_BASE = {self.knowledge_base}'


def validate_schema(df, metadata_dict):
    """
    Validate the schema of the columns of the dataframe according to the metadata_dict.

    If a column is not present in the dataframe, it is ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to be validated

    metadata_dict: dict,
        dictionary with the schema of the columns

    Returns
    -------
    errors: list
       list of errors found in the validation
    """
    print('Starting validation')
    errors = []

    for col, metadata in metadata_dict.items():
        if col in df.columns:
            print(f'{col}: {metadata.schema}')
            is_int = 'type' in metadata_dict[col].schema and metadata_dict[col].schema['type'] == 'integer'
            for i in range(len(df)):
                try:
                    # trick to solve validation error due to integer data types casted to float by Python
                    jsonschema.validate(int(df.iloc[i][col]) if is_int else df.iloc[i][col], metadata.schema)
                except jsonschema.ValidationError as e:
                    print(f'Error in {col} at index {i}: {e}')
                    errors.append((col, i, e))
    print('End of validation')
    return errors
