import pandas as pd

from ..util import dataframe_like
from ..io import load_dataframe
from .polars_dataframe import is_polars_dataframe, is_polars_lazyframe, wrangle_polars_dataframe, pandas_to_polars, polars_to_pandas
from ..util.lazy_imports import get_polars


def is_dataframe(x):
    """
    Determine if an object (or file) is a DataFrame (pandas or Polars)

    Parameters
    ----------
    :param x: the object (or a file path)

    Returns
    -------
    :return: True if the object is a DataFrame (pandas or Polars) or points to a file that can be loaded as a DataFrame,
    and False otherwise.
    """
    # Check for pandas DataFrames
    if type(x).__module__ in ['pandas.core.frame', 'modin.pandas.dataframe']:
        return True
    
    # Check for Polars DataFrames
    if is_polars_dataframe(x) or is_polars_lazyframe(x):
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
    """
    Determine if an object (or file) is a MultiIndex DataFrame-- i.e., a DataFrame with a multi-level index

    Parameters
    ----------
    :param x: the object (or file path)

    Returns
    -------
    :return: True if the object is a MultiIndex DataFrame (or points to a file that can be loaded as a
    MultiIndex DataFrame), and False otherwise.
    """
    return is_dataframe(x) and ('indexes.multi' in type(x.index).__module__)


def wrangle_dataframe(data, return_model=False, backend=None, **kwargs):
    """
    Turn a (potentially messy) DataFrame into a (potentially cleaner) DataFrame

    Parameters
    ----------
    :param data: a DataFrame (pandas or Polars), dataframe-like object, or a file path that points to a file that can be 
      loaded as a DataFrame
    :param return_model: if True, return a function for turning the ("messy") DataFrame into a "clean" DataFrame, along with
      the cleaned DataFrame.  Otherwise (if False), just return the cleaned DataFrame.  Default: False
    :param backend: str, optional
        The DataFrame backend to use ('pandas' or 'polars'). If None, preserves the input type
    :param kwargs: passed to the DataFrame "wrangling" model (default: the constructor for pandas.DataFrame or polars.DataFrame)

    Returns
    -------
    :return: The "wrangled" DataFrame (if return_model is False), or the DataFrame plus a "model" for cleaning
      DataFrames (if return_model is True).

    Examples
    --------
    >>> import pandas as pd
    >>> import datawrangler as dw
    >>> # Wrangle pandas DataFrame, preserving type
    >>> df_pandas = pd.DataFrame({'A': [1, 2, 3]})
    >>> cleaned_pandas = dw.wrangle(df_pandas)
    >>> # Convert pandas DataFrame to Polars
    >>> df_polars = dw.wrangle(df_pandas, backend='polars')
    >>> # Load and wrangle from file
    >>> df_from_file = dw.wrangle('data.csv')
    """
    load_kwargs = kwargs.pop('load_kwargs', {})

    # Handle Polars DataFrames
    if is_polars_dataframe(data) or is_polars_lazyframe(data):
        # If it's already a Polars DataFrame, route to Polars handler
        if backend == 'pandas':
            # Convert to pandas if explicitly requested
            data = polars_to_pandas(data)
            data = load_dataframe(data, **load_kwargs)
        else:
            # Keep as Polars
            return wrangle_polars_dataframe(data, return_model=return_model, **kwargs)
    else:
        # Load as pandas DataFrame first
        data = load_dataframe(data, **load_kwargs)
        
        # Convert to Polars if requested
        if backend == 'polars':
            data = pandas_to_polars(data)
            return wrangle_polars_dataframe(data, return_model=return_model, **kwargs)
    
    # Handle pandas DataFrames
    model = kwargs.pop('model', None)
    if model is None:
        model = {'model': pd.DataFrame, 'args': [], 'kwargs': kwargs}
    elif type(model) is not dict:
        model = {'model': model, 'args': [], 'kwargs': kwargs}

    wrangled = model['model'](data, *model['args'], **model['kwargs'])

    if return_model:
        return wrangled, model
    return wrangled
