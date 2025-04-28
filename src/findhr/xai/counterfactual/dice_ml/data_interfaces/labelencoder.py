import array
import itertools
import warnings
from collections import defaultdict
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import column_or_1d
from sklearn.utils._array_api import _setdiff1d, device, get_namespace
from sklearn.utils._encode import _encode, _unique, _map_to_integer, _get_counts, _extract_missing, _searchsorted, is_scalar_nan
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.sparsefuncs import min_max_axis
from sklearn.utils.validation import _num_samples, check_array, check_is_fitted


def _unique(values, *, return_inverse=False, return_counts=False):
    """Helper function to find unique values with support for python objects.

    Uses pure python method for object dtype, and numpy method for
    all other dtypes.

    Parameters
    ----------
    values : ndarray
        Values to check for unknowns.

    return_inverse : bool, default=False
        If True, also return the indices of the unique values.

    return_counts : bool, default=False
        If True, also return the number of times each unique item appears in
        values.

    Returns
    -------
    unique : ndarray
        The sorted unique values.

    unique_inverse : ndarray
        The indices to reconstruct the original array from the unique array.
        Only provided if `return_inverse` is True.

    unique_counts : ndarray
        The number of times each of the unique values comes up in the original
        array. Only provided if `return_counts` is True.
    """
    if values.dtype == object:
        return _unique_python(
            values, return_inverse=return_inverse, return_counts=return_counts
        )
    # numerical
    return _unique_np(
        values, return_inverse=return_inverse, return_counts=return_counts
    )

def _unique_python(values, *, return_inverse, return_counts):
    # Only used in `_uniques`, see docstring there for details
    try:
        uniques_set = set(values)
        uniques_set, missing_values = _extract_missing(uniques_set)

        uniques_list = sorted(uniques_set)
        uniques_list.extend(missing_values.to_list())
        # NEW:
        uniques = np.empty(len(uniques_list), dtype=values.dtype)
        uniques[:] = uniques_list
        # OLD:
        # uniques = np.array(uniques, dtype=values.dtype)
    except TypeError:
        types = sorted(t.__qualname__ for t in set(type(v) for v in values))
        raise TypeError(
            "Encoders require their input argument must be uniformly "
            f"strings or numbers. Got {types}"
        )
    ret = (uniques,)

    if return_inverse:
        ret += (_map_to_integer(values, uniques),)

    if return_counts:
        ret += (_get_counts(values, uniques),)

    return ret[0].flatten() if len(ret) == 1 else ret

def _unique_np(values, return_inverse=False, return_counts=False):
    """Helper function to find unique values for numpy arrays that correctly
    accounts for nans. See `_unique` documentation for details."""
    xp, _ = get_namespace(values)

    inverse, counts = None, None

    if return_inverse and return_counts:
        uniques, _, inverse, counts = xp.unique_all(values)
    elif return_inverse:
        uniques, inverse = xp.unique_inverse(values)
    elif return_counts:
        uniques, counts = xp.unique_counts(values)
    else:
        uniques = xp.unique_values(values)

    # np.unique will have duplicate missing values at the end of `uniques`
    # here we clip the nans and remove it from uniques
    if uniques.size and is_scalar_nan(uniques[-1]):
        nan_idx = _searchsorted(uniques, xp.nan, xp=xp)
        uniques = uniques[: nan_idx + 1]
        if return_inverse:
            inverse[inverse > nan_idx] = nan_idx

        if return_counts:
            counts[nan_idx] = xp.sum(counts[nan_idx:])
            counts = counts[: nan_idx + 1]

    ret = (uniques,)

    if return_inverse:
        ret += (inverse,)

    if return_counts:
        ret += (counts,)

    return ret[0] if len(ret) == 1 else ret


class LabelEncoder(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Encode target labels with value between 0 and n_classes-1.

    This transformer should be used to encode target values, *i.e.* `y`, and
    not the input `X`.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    .. versionadded:: 0.12

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    See Also
    --------
    OrdinalEncoder : Encode categorical features using an ordinal encoding
        scheme.
    OneHotEncoder : Encode categorical features as a one-hot numeric array.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from sklearn.preprocessing import LabelEncoder
    >>> le = LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    [np.str_('amsterdam'), np.str_('paris'), np.str_('tokyo')]
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    [np.str_('tokyo'), np.str_('tokyo'), np.str_('paris')]
    """

    def fit(self, y):
        """Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
            Fitted label encoder.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = _unique(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = _unique(y, return_inverse=True)
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Labels as normalized encodings.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(y)
        y = column_or_1d(y, dtype=self.classes_.dtype, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return xp.asarray([])

        return _encode(y, uniques=self.classes_)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Original encoding.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(y)
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return xp.asarray([])

        diff = _setdiff1d(
            ar1=y,
            ar2=xp.arange(self.classes_.shape[0], device=device(y)),
            xp=xp,
        )
        if diff.shape[0]:
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = xp.asarray(y)
        return xp.take(self.classes_, y, axis=0)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.array_api_support = True
        tags.input_tags.two_d_array = False
        tags.target_tags.one_d_labels = True
        return tags
