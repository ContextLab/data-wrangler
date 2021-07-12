import pandas as pd

from ..util import dataframe_like
from ..io import load_dataframe


def is_dataframe(x):
    if type(x).__module__ in ['pandas.core.frame', 'modin.pandas.dataframe']:
        return True
    else:
        if dataframe_like(x):
            return True

        # noinspection PyBroadException
        try:
            data = load_dataframe(x)
            return data is not None
        except:
            return False


def is_multiindex_dataframe(x):
    return is_dataframe(x) and ('indexes.multi' in type(x.index).__module__)


def wrangle_dataframe(data, return_model=False, **kwargs):
    if return_model:
        return pd.DataFrame(data, **kwargs), {'model': pd.DataFrame, 'args': [], 'kwargs': kwargs}
    return pd.DataFrame(data, **kwargs)
