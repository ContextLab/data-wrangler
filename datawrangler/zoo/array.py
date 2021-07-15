import pandas as pd
import numpy as np
import six
import os
from ..io import load


def is_array(x):
    if (not ('str' in str(type(x)))) and (type(x).__module__ == 'numpy'):
        return True
    else:
        # noinspection PyBroadException
        try:
            if is_array(load(x)):
                return True
        except:
            if np.isscalar(x) or type(x) == list:
                return True
    return False


def wrangle_array(data, return_model=False, **kwargs):
    def stacker(x):
        while x.ndim >= 3:
            last_dim = x.ndim - 1
            x = np.concatenate(np.split(x, x.shape[last_dim], axis=last_dim), axis=last_dim-1)
            x = np.squeeze(x)
        return x

    if np.isscalar(data) or (type(data) is list):
        data = np.array(data)
    elif (type(data) in six.string_types) and os.path.exists(data) and is_array(data):
        data = load(data)

    if ('sparse' in str(type(data))) and hasattr(data, 'toarray'):
        data = data.toarray()

    data = stacker(np.atleast_2d(data))

    if return_model:
        return pd.DataFrame(data, **kwargs), {'model': pd.DataFrame, 'args': [], 'kwargs': kwargs}
    return pd.DataFrame(data, **kwargs)
