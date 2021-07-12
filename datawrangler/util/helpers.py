import numpy as np
import pandas as pd
import six
import warnings

from ..io.io import get_extension
from ..format.array import is_array


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


def load_dataframe(x, extension=None, **kwargs):
    if type(x) in six.string_types:
        if extension is None:
            extension = get_extension(x)

        # built-in pandas parsers support both local and remote loading
        if extension == 'csv':
            return pd.read_csv(x, **kwargs)
        elif extension in ['xls', 'xlsx']:
            return pd.read_excel(x, **kwargs)
        elif extension == 'json':
            return pd.read_json(x, **kwargs)
        elif extension == 'html':
            return pd.read_html(x, **kwargs)
        elif extension == 'xml':
            return pd.read_xml(x, **kwargs)
        elif extension == 'hdf':
            return pd.read_hdf(x, **kwargs)
        elif extension == 'feather':
            return pd.read_feather(x, **kwargs)
        elif extension == 'parquet':
            return pd.read_parquet(x, **kwargs)
        elif extension == 'orc':
            return pd.read_orc(x, **kwargs)
        elif extension == 'sas':
            return pd.read_sas(x, **kwargs)
        elif extension == 'spss':
            return pd.read_spss(x, **kwargs)
        elif extension == 'sql':
            return pd.read_sql(x, **kwargs)
        elif extension == 'gbq':
            return pd.read_gbq(x, **kwargs)
        elif extension == 'stata':
            return pd.read_stata(x, **kwargs)
        elif extension == 'pkl':
            return pd.read_pickle(x, **kwargs)
        else:
            warnings.warn(f'cannot determine filetype: {x}')
            return None
    elif dataframe_like(x):
        return x
    else:
        return None
