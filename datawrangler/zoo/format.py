import six
import numpy as np
import pandas as pd

from .dataframe import is_dataframe, is_multiindex_dataframe, wrangle_dataframe
from .array import is_array, wrangle_array
from .text import is_text, wrangle_text
from .null import is_null, wrangle_null
from ..util import array_like, depth
from ..core import update_dict, get_default_options

# the order matters: if earlier checks pass, later checks will not run.
# the list specifies the priority of converting to the given data types.
format_checkers = eval(get_default_options()['supported_formats']['types'])


def wrangle(x, return_dtype=False, **kwargs):
    """
    Turn messy data into clean data

    Parameters
    ----------
    :param x: data in any format (text, numpy arrays, pandas dataframes, or a mixed list (or nested lists) of those
      types).  The following datatypes are supported:
        - Numpy Arrays, array-like objects, or paths to files that store array-like objects
        - Pandas DataFrames, dataframe-like objects, or paths to files that store dataframe-like objects
        - Images, or paths to files that store images
        - Text, or paths to plain text files
        - Mixed lists of the above
    :param return_dtype: if True, also return the auto-detected datatype(s) of each dataset you wrangle
    :param kwargs: used to control how data are wrangled (e.g., if you don't want to use the default options for each
      data type):
        - array_kwargs: passed to the datawrangler.zoo.array.wrangle_array function to control how arrays are handled
        - dataframe_kwargs: passed to the datawrangler.zoo.dataframe.wrangle_dataframe function to control how
          dataframes are handled
        - image_kwargs: passed to the datawrangler.zoo.image.wrangle_image function to control how images are handled
        - text_kwargs: passed to the datawrangler.zoo.text.wrangle_text function to control how text data are handled
      any other keyword arguments are passed to *all* of the wrangle functions.

    Returns
    -------
    :return: a DataFrame, or a list of DataFrames, containing the wrangled data
    """

    deep_kwargs = {}
    for f in format_checkers:
        deep_kwargs[f] = kwargs.pop(f'{f}_kwargs', {})

    for f in format_checkers:
        deep_kwargs[f] = update_dict(kwargs, deep_kwargs[f])

    pre_fit = {f: False for f in format_checkers}

    # noinspection PyUnusedLocal,PyShadowingNames
    def to_dataframe(y):
        dtype = None
        wrangled = pd.DataFrame()
        for fc in format_checkers:
            if eval(f'is_{fc}(y)'):
                wrangler = eval(f'wrangle_{fc}')
                if not pre_fit[fc]:
                    return_model = ('return_model' in deep_kwargs[fc].keys()) and deep_kwargs[fc]['return_model']
                    deep_kwargs[fc]['return_model'] = True
                    wrangled, model = wrangler(y, **deep_kwargs[fc])

                    deep_kwargs[fc]['model'] = model
                    deep_kwargs[fc]['return_model'] = return_model
                    pre_fit[fc] = True

                    if return_model:
                        wrangled = [wrangled, model]
                else:
                    wrangled = wrangler(y, **deep_kwargs[fc])
                dtype = fc
                break
        return wrangled, dtype

    if ((not is_text(x)) and (type(x) == list)) or (is_text(x) and (type(x) == list) and (depth(x) > 1)):
        dfs = [to_dataframe(i) for i in x]
        wrangled = [d[0] for d in dfs]
        dtypes = [d[1] for d in dfs]
    else:
        wrangled, dtypes = to_dataframe(x)

    if return_dtype:
        return wrangled, dtypes
    else:
        return wrangled
