import numpy as np
import pandas as pd
from ..util.lazy_imports import get_polars


def is_null(data):
    """
    Test whether an object is None or empty.

    Parameters
    ----------
    :param data: the to-be-tested object

    Returns
    -------
    :return: True if the object is None or empty and False otherwise.

    """
    # noinspection PyBroadException
    try:
        if np.iterable(data):
            return all([is_null(d) for d in data])
        return (data is None) or (len(data) == 0)
    except:
        return False


def wrangle_null(data, return_model=False, backend=None, model=None):
    """
    Turn a null object (None or empty) into an empty DataFrame (pandas or Polars).

    Parameters
    ----------
    :param data: the to-be-wrangled null object
    :param return_model: if True, return a model (and its arguments and keyword arguments) for turning the null object
      into a DataFrame.
    :param backend: str, optional
        The DataFrame backend to use ('pandas' or 'polars'). If None, uses the default backend (pandas)
    :param model: a function or constructor that will generate an empty DataFrame.  This can also be specified as a
      dictionary with the following fields:
        - 'model': a function or constructor
        - 'args': a list of unnamed arguments to be passed to the given function or constructor
        - 'kwargs': a dictionary of named arguments to be passed to the given function or constructor
      If None, defaults to pandas.DataFrame or polars.DataFrame based on backend.

    Returns
    -------
    :return: an empty DataFrame (pandas or Polars based on backend)

    Examples
    --------
    >>> import datawrangler as dw
    >>> # Create empty pandas DataFrame (default)
    >>> df_pandas = dw.wrangle(None)
    >>> # Create empty Polars DataFrame
    >>> df_polars = dw.wrangle(None, backend='polars')
    """

    # Determine default model based on backend
    if model is None:
        if backend == 'polars':
            pl = get_polars()
            default_model = pl.DataFrame
        else:
            default_model = pd.DataFrame
        model = default_model
    
    if type(model) is not dict:
        model = {'model': model, 'args': [], 'kwargs': {}}

    x = model['model'](*model['args'], **model['kwargs'])

    if return_model:
        return x, model
    return x
