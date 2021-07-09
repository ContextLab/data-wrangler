import numpy as np
import pandas as pd
import modin
import six
import os
from array import is_array

from ..io import get_extension, load_remote, load_dataframe
from ..helpers import dataframe_like


def is_dataframe(x):
    if type(x).__module__ in ['pandas.core.frame', 'modin.pandas.dataframe']:
        return True
    else:
        if dataframe_like(x):
            return True
        data = load_dataframe(x)
        return data is not None


def is_multiindex_dataframe(x):
    return is_dataframe(x) and ('indexes.multi' in type(x.index).__module__)


def wrangle_dataframe(data, return_model=False, **kwargs):
    if return_model:
        return pd.DataFrame(data, **kwargs), {'model': pd.DataFrame, 'args': [], 'kwargs': kwargs}
    return pd.DataFrame(data, **kwargs)
