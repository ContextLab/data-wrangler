import six
import numpy as np
import pandas as pd

from .dataframe import is_dataframe, is_multiindex_dataframe, wrangle_dataframe
from .array import is_array, wrangle_array
from .image import is_image, wrangle_image
from .text import is_text, wrangle_text
from .null import is_null, wrangle_null
from ..util import array_like

# the order matters: if earlier checks pass, later checks will not run.
# the list specifies the priority of converting to the given data types.
format_checkers = ['dataframe', 'array', 'image', 'text', 'null']


def wrangle(x, return_dtype=False, **kwargs):
    """
    INPUTS
    x: data in any format (text, numpy arrays, pandas dataframes, or a mixed list (or nested lists) of those types)

    OUTPUTS
    a list of pandas dataframes
    """

    deep_kwargs = {}
    for f in format_checkers:
        deep_kwargs[f] = {}
        if f'{f}_kwargs' in kwargs.keys():
            deep_kwargs[f'{f}_kwargs'] = kwargs.pop(f'{f}_kwargs', None)

    # noinspection PyUnusedLocal,PyShadowingNames
    def to_dataframe(y, deep_kwargs, **kwargs):
        dtype = None
        wrangled = pd.DataFrame()
        for fc in format_checkers:
            if eval(f'is_{fc}(y)'):
                wrangled = eval(f'wrangle_{fc}(y, **deep_kwargs["{fc}"], **kwargs)')
                dtype = fc
        return wrangled, dtype

    if type(x) == list:
        dfs = [to_dataframe(i, deep_kwargs, **kwargs) for i in x]
        wrangled = [dfs[0] for _ in dfs]
        dtypes = [dfs[1] for _ in dfs]
    else:
        wrangled, dtypes = to_dataframe(x)

    if return_dtype:
        return wrangled, dtypes
    else:
        return wrangled
