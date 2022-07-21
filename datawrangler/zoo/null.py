import numpy as np
import pandas as pd


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


def wrangle_null(data, return_model=False, model=pd.DataFrame):
    """
    Turn a null object (None or empty) into an empty DataFrame.

    Parameters
    ----------
    :param data: the to-be-wrangled null object
    :param return_model: if True, return a model (and its arguments and keyword arguments) for turning the null object
      into a DataFrame.
    :param model: a function or constructor that will generate an empty DataFrame.  This can also be specified as a
      dictionary with the following fields:
        - 'model': a function or constructor
        - 'args': a list of unnamed arguments to be passed to the given function or constructor
        - 'kwargs': a dictionary of named arguments to be passed to the given function or constructor

    Returns
    -------
    :return: an empty DataFrame
    """

    if type(model) is not dict:
        model = {'model': model, 'args': [], 'kwargs': {}}

    x = model['model'](*model['args'], **model['kwargs'])

    if return_model:
        return x, model
    return x
