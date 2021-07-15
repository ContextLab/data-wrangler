import numpy as np
import pandas as pd


def is_null(data):
    # noinspection PyBroadException
    try:
        if np.iterable(data):
            return all([is_null(d) for d in data])
        return (data is None) or (len(data) == 0)
    except:
        return False


def wrangle_null(data, return_model=False, **kwargs):
    x = pd.DataFrame()

    model = kwargs.pop('model', pd.DataFrame)
    if type(model) is not dict:
        model = {'model': model, 'args': [], 'kwargs': kwargs}

    if return_model:
        return x, model
    return x
