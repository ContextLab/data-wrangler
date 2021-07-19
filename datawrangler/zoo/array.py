import pandas as pd
import numpy as np
import six
import os
from ..io import load
from ..core.configurator import update_dict


def is_number(x):
    """
    Internal function-- return whether an object is a numerical scalar

    Parameters
    ----------
    :param x: the object to test

    Returns
    -------
    :return: True of x is a real or complex scalar and False otherwise
    """
    if np.isscalar(x):
        return np.isreal(x) or np.iscomplex(x)  # exclude single characters (non-numeric)
    if type(x) is list:
        return all([is_number(i) for i in x])
    return False


def is_array(x):
    """
    Return True if and only if is an Array, or a file that can be loaded into an Array.

    Parameters
    ----------
    :param x: an object, file path or URL

    Returns
    -------
    :return: whether (or not) x is an array (or if it points to an array)
    """
    if (not ('str' in str(type(x)))) and (type(x).__module__ == 'numpy'):
        return True
    else:
        # noinspection PyBroadException
        try:
            if is_array(load(x)):
                return True
        except:
            if type(x) == list:
                return all([is_array(i) for i in x])
            elif is_number(x):
                return True

    return False


def wrangle_array(data, return_model=False, **kwargs):
    """
    Turn an Array into a Pandas DataFrame

    Parameters
    ----------
    :param data: an Array (or path to an Array)
    :param return_model: if True, return a function for casting an Array into a DataFrame (along with the resulting
      DataFrame).  Default: False
    :param kwargs: a list of keyword arguments:
       - 'model': a callable function or constructor, or a dictionary containing the following keys:
         - 'model': a callable function or constructor
         - 'args': a list of arguments to pass to the function (in addition to data)
         - 'kwargs': a list of keyword arguments to pass to the function
         default: pandas.DataFrame
       - all other keyword arguments are passed to the model (or constructor).  These can be used to change how the
         DataFrame is created (e.g., passing columns=['one', 'two', 'three'] will change the column names of the
         resulting DataFrame, assuming the "model" is pandas.DataFrame).

    Returns
    -------
    :return: The resulting DataFrame
    """
    def stacker(x):
        while x.ndim >= 3:
            last_dim = x.ndim - 1
            x = np.concatenate(np.split(x, x.shape[last_dim], axis=last_dim), axis=last_dim-1)
            x = np.squeeze(x)
        return x

    if is_number(data):
        data = np.array(data)
    elif (type(data) in six.string_types) and os.path.exists(data) and is_array(data):
        data = load(data)

    if ('sparse' in str(type(data))) and hasattr(data, 'toarray'):
        data = data.toarray()

    data = stacker(np.atleast_2d(data))

    model = kwargs.pop('model', pd.DataFrame)
    if type(model) is dict:
        # noinspection PyArgumentList
        assert all([k in model.keys() for k in ['model', 'args', 'kwargs']]), ValueError(f'Invalid model: {model}')
        model_args = model['args']
        model_kwargs = update_dict(model['kwargs'], kwargs)
        model = model['model']
    else:
        model_args = []
        model_kwargs = kwargs

    wrangled = model(data, *model_args, **model_kwargs)

    if return_model:
        return wrangled, {'model': model, 'args': model_args, 'kwargs': model_kwargs}
    return wrangled
