import numpy as np
import pandas as pd
import six
import warnings
import os

from ..io import load
from ..io.extension_handler import get_extension
from ..zoo.array import is_array


def btwn(x, a, b):
    """
    Test whether the values of an array (or scalar) are between the given bounds (inclusive)

    Parameters
    ----------
    :param x: a numeric scalar or Array
    :param a: lower bound-- a numeric scalar or an Array of the same shape as x
    :param b: upper bound-- a numeric scalar or an Array of the same shape as x

    Returns
    -------
    :return: True if the values in the given Array are between the given bounds (inclusive), and False otherwise.

    """
    assert np.isscalar(a), ValueError(f'lower limit must be scalar (given: {type(a)}')
    assert np.isscalar(b), ValueError(f'upper limit must be scalar (given: {type(b)}')

    if b < a:
        return btwn(x, b, a)

    return np.all(x >= a) and np.all(x <= b)


def dataframe_like(x, debug=False):
    """
    Determine whether an object can be treated as a Pandas DataFrame for wrangling purposes

    Parameters
    ----------
    :param x: the object
    :param debug: internal flag (default: False) that prints out why an object is *not* dataframe like

    Returns
    -------
    :return: True (if the object can be treated like a DataFrame) or False otherwise
    """
    required_attributes = ['values', 'index', 'columns', 'shape', 'stack', 'unstack', 'loc', 'iloc', 'size', 'copy',
                           'head', 'tail', 'items', 'iteritems', 'keys', 'iterrows', 'itertuples',
                           'where', 'query', 'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod',
                           'pow', 'dot', 'radd', 'rsub', 'rmul', 'rdiv', 'rtruediv', 'rfloordiv', 'rmod', 'rpow',
                           'lt', 'gt', 'le', 'ge', 'ne', 'eq', 'apply', 'groupby', 'rolling', 'expanding', 'abs',
                           'filter', 'drop', 'drop_duplicates', 'backfill', 'bfill', 'ffill', 'fillna', 'interpolate',
                           'pad', 'droplevel', 'pivot', 'pivot_table', 'squeeze', 'melt', 'join', 'merge']
    for r in required_attributes:
        if not hasattr(x, r):
            if debug:
                print(f'missing method: {r}')
            return False
    return True


def array_like(x, force_literal=False):
    """
    Determine whether an object can be treated as a Numpy Array for wrangling purposes

    Parameters
    ----------
    :param x: the object
    :param force_literal: specify whether strings should be interpreted strictly (if force_literal == True) or whether
      they may refer to files or URLs (if force_literal == False).  Default: False

    Returns
    -------
    :return: True (if the object can be treated like an Array) or False otherwise
    """
    if type(x) is str:
        if force_literal:
            return False
        else:
            return array_like(load(x), force_literal=True)

    return is_array(x) or dataframe_like(x) or (type(x) in [list, np.array, np.ndarray, pd.Series, pd.DataFrame])


def depth(x):
    """
    Determine the maximum depth of a list or array (i.e., maximum amount of nesting, across all elements)

    Parameters
    ----------
    :param x: an array-like object

    Returns
    -------
    :return: The depth of the object
    """
    if array_like(x):
        if np.isscalar(x) or (len(x) == 0):
            return 0
        else:
            return 1 + np.max([depth(i) for i in x])
    else:
        return 0
