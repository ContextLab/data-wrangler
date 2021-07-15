import pandas as pd

from ..util import dataframe_like
from ..io import load_dataframe


def is_dataframe(x):
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
    return is_dataframe(x) and ('indexes.multi' in type(x.index).__module__)


def wrangle_dataframe(data, return_model=False, **kwargs):
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
