import pandas as pd
import numpy as np
import six
import os
from ..io import load
from ..core.configurator import update_dict


def is_number(x):
    if np.isscalar(x):
        return np.isreal(x) or np.iscomplex(x)  # exclude single characters (non-numeric)
    if type(x) is list:
        return all([is_number(i) for i in x])
    return False


def is_array(x):
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
