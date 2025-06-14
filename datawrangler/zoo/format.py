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


def wrangle(x, return_dtype=False, backend=None, **kwargs):
    """
    Turn messy data into clean DataFrames (pandas or Polars)

    Automatically detects and converts various data types into consistent DataFrame format.
    Specializes in text processing using modern NLP models and handles mixed data types.

    Parameters
    ----------
    :param x: data in any format. Supported datatypes:
        - Numpy Arrays, array-like objects, or paths to files that store array-like objects
        - DataFrames (pandas or Polars), dataframe-like objects, or paths to files that store dataframe-like objects  
        - Polars LazyFrames
        - Text strings, lists of strings, or paths to plain text files
        - Mixed lists or nested lists of the above types
    :param return_dtype: if True, also return the auto-detected datatype(s) of each dataset. Default: False
    :param backend: str, optional
        The DataFrame backend to use ('pandas' or 'polars'). If None, uses the default backend (pandas)
    :param kwargs: control how data are wrangled:
        - array_kwargs: passed to wrangle_array function to control how arrays are handled
        - dataframe_kwargs: passed to wrangle_dataframe function to control how dataframes are handled
        - text_kwargs: passed to wrangle_text function to control how text data are handled
            Common text_kwargs options (simplified API):
            - {'model': 'all-MiniLM-L6-v2'} for sentence-transformers
            - {'model': 'CountVectorizer'} for sklearn text vectorization
            - {'model': ['CountVectorizer', 'LatentDirichletAllocation']} for sklearn pipeline
            Also supports full dict format for advanced configuration:
            - {'model': {'model': 'all-MiniLM-L6-v2', 'args': [], 'kwargs': {}}}
        Any other keyword arguments are passed to all wrangle functions.

    Returns
    -------
    :return: a DataFrame (pandas or Polars), or a list of DataFrames, containing the wrangled data
    
    Examples
    --------
    >>> import datawrangler as dw
    >>> # Convert array to pandas DataFrame (default)
    >>> df = dw.wrangle([1, 2, 3])
    >>> # Convert array to Polars DataFrame
    >>> df_polars = dw.wrangle([1, 2, 3], backend='polars')
    >>> # Process text with sentence-transformers model
    >>> text_df = dw.wrangle(["Hello", "World"], text_kwargs={'model': 'all-MiniLM-L6-v2'})
    >>> # Handle mixed data types and return detected types
    >>> mixed_df, dtypes = dw.wrangle([df, text_df], return_dtype=True)
    """

    deep_kwargs = {}
    for f in format_checkers:
        deep_kwargs[f] = kwargs.pop(f'{f}_kwargs', {})

    for f in format_checkers:
        deep_kwargs[f] = update_dict(kwargs, deep_kwargs[f])
        # Pass backend parameter to all wranglers
        if backend is not None:
            deep_kwargs[f]['backend'] = backend

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
