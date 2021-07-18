import pandas as pd

from ..util import dataframe_like
from ..io import load_dataframe


def is_dataframe(x):
    """
    Determine if an object (or file) is a DataFrame

    Parameters
    ----------
    :param x: the object (or a file path)

    Returns
    -------
    :return: True if the object is a DataFrame (or points to a file that can be loaded into Pandas as a DataFrame), and
    False otherwise.
    """
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
    """
    Determine if an object (or file) is a MultiIndex DataFrame-- i.e., a DataFrame with a multi-level index

    Parameters
    ----------
    :param x: the object (or file path)

    Returns
    -------
    :return: True if the object is a MultiIndex DataFrame (or points to a file that can be loaded into Pandas as a
    MultiIndex DataFrame), and False otherwise.
    """
    return is_dataframe(x) and ('indexes.multi' in type(x.index).__module__)


def wrangle_dataframe(data, return_model=False, **kwargs):
    """
    Turn a (potentially messy) DataFrame into a (potentially cleaner) DataFrame

    Parameters
    ----------
    :param data: a DataFrame, dataframe-like object, or a file path that points to a file that can be loaded into Pandas as a
      DataFrame
    :param return_model: if True, return a function for turning the ("messy") DataFrame into a "clean" DataFrame, along with
      the cleaned DataFrame.  Otherwise (if False), just return the cleaned DataFrame.  Default: False
    :param kwargs: passed to the DataFrame "wrangling" model (default: the constructor for pd.DataFrame)

    Returns
    -------
    :return: The "wrangled" DataFrame (if return_model is False), or the DataFrame plus a "model" for cleaning
      DataFrames (if return_model is True).
    """
    load_kwargs = kwargs.pop('load_kwargs', {})

    data = load_dataframe(data, **load_kwargs)
    model = kwargs.pop('model', None)
    if model is None:
        model = {'model': pd.DataFrame, 'args': [], 'kwargs': kwargs}
    elif type(model) is not dict:
        model = {'model': model, 'args': [], 'kwargs': kwargs}

    wrangled = model['model'](data, *model['args'], **model['kwargs'])

    if return_model:
        return wrangled, model
    return wrangled
