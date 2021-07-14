import numpy as np
import pandas as pd
import six
import warnings

from ..io.extension_handler import get_extension
from ..zoo.array import is_array


def btwn(x, a, b):
    assert np.isscalar(a), ValueError(f'lower limit must be scalar (given: {type(a)}')
    assert np.isscalar(b), ValueError(f'upper limit must be scalar (given: {type(b)}')

    if b < a:
        return btwn(x, b, a)

    return np.all(x >= a) and np.all(b <= b)


def dataframe_like(x, debug=False):
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


def array_like(x):
    return is_array(x) or dataframe_like(x) or (type(x) in [list, np.array, np.ndarray, pd.Series, pd.DataFrame])


def depth(x):
    if array_like(x):
        if len(x) == 0:
            return 1
        else:
            return 1 + np.max([depth(i) for i in x])
    else:
        return 0
