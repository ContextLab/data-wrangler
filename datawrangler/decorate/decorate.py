import warnings
import functools
import numpy as np
import pandas as pd
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
import sklearn.feature_extraction.text as text
import sklearn.mixture as mixture

from sklearn.experimental import enable_iterative_imputer
import sklearn.impute as impute

from ..zoo import wrangle
from ..zoo.text import is_sklearn_model
from ..zoo.dataframe import is_dataframe, is_multiindex_dataframe
from ..zoo.array import is_array
from ..core import get_default_options, apply_defaults, update_dict
from ..util.helpers import depth


defaults = get_default_options()
format_checkers = defaults['supported_formats']['types']


# import all model-like classes within a sklearn-like module; return a list of model names
# note: this is NOT a decorator-- it's a helper function used to seed functions for the module_checker decorator
def import_sklearn_models(module):
    models = [d for d in dir(module) if hasattr(getattr(module, d), 'fit_transform')]
    for m in models:
        exec(f'from {module.__name__} import {m}', globals())
    return models


def get_sklearn_model(x):
    if is_sklearn_model(x):
        return x  # already a valid model
    elif type(x) is dict:
        if hasattr(x, 'model'):
            return get_sklearn_model(x['model'])
        else:
            return None
    elif type(x) is str:
        # noinspection PyBroadException
        try:
            return get_sklearn_model(eval(x))
        except:
            pass
    return None


# FIXME: this code is partially duplicated from zoo.text.apply_text-model
def apply_sklearn_model(model, data, *args, mode='fit_transform', return_model=False, **kwargs):
    assert mode in ['fit', 'transform', 'fit_transform']
    if type(model) is list:
        models = []
        for i, m in enumerate(model):
            if (i < len(model) - 1) and ('transform' not in mode):
                temp_mode = 'fit_transform'
            else:
                temp_mode = mode

            data, m = apply_sklearn_model(m, data, *args, mode=temp_mode, return_model=True, **kwargs)
            models.append(m)

        if return_model:
            return data, models
        else:
            return data
    elif type(model) is dict:
        assert all([k in model.keys() for k in ['model', 'args', 'kwargs']]), ValueError(f'invalid model: {model}')
        return apply_sklearn_model(model['model'], data, *[*model['args'], *args], mode=mode, return_model=return_model,
                                   **update_dict(model['kwargs'], kwargs))

    model = get_sklearn_model(model)
    if model is None:
        raise RuntimeError(f'unsupported model: {model}')
    model = apply_defaults(model)(*args, **kwargs)

    m = getattr(model, mode)
    transformed_data = m(data)
    if return_model:
        return transformed_data, {'model': model, 'args': args, 'kwargs': kwargs}
    return transformed_data


reduce_models = ['UMAP']
reduce_models.extend(import_sklearn_models(decomposition))
reduce_models.extend(import_sklearn_models(manifold))

text_vectorizers = import_sklearn_models(text)

impute_models = import_sklearn_models(impute)

# source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
interpolation_models = ['linear', 'time', 'index', 'pad', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline',
                        'barycentric', 'polynomial']


# make a function work for either a single object or a list of objects by calling the function on each element
def list_generalizer(f):
    @functools.wraps(f)
    def wrapped(data, **kwargs):
        if type(data) == list:
            return [f(d, **kwargs) for d in data]
        else:
            return f(data, **kwargs)

    return wrapped


# coerce the data passed into the function into a pandas dataframe or a list of dataframes
def funnel(f):
    @functools.wraps(f)
    def wrapped(data, *args, **kwargs):
        wrangle_kwargs = kwargs.pop('wrangle_kwargs', {})
        for fc in format_checkers:
            wrangle_kwargs[f'{fc}_kwargs'] = kwargs.pop(f'{fc}_kwargs', {})

        return f(wrangle(data, **wrangle_kwargs), *args, **kwargs)

    return wrapped


# fill in missing data by imputing and/or interpolating
def interpolate(f):
    @funnel
    def fill_missing(data, return_model=False, **kwargs):
        impute_kwargs = kwargs.pop('impute_kwargs', {})

        if impute_kwargs:
            model = impute_kwargs.pop('model', defaults['impute']['model'])
            imputed_data, model = apply_sklearn_model(model, data, return_model=True, **impute_kwargs)
            data = pd.DataFrame(data=imputed_data, index=data.index, columns=data.columns)
        else:
            model = None

        if kwargs:
            kwargs = update_dict(defaults['interpolate'], kwargs)
            data = data.interpolate(**kwargs)

        if return_model:
            return data, {'model': model, 'args': [], 'kwargs': kwargs}
        else:
            return data

    @functools.wraps(f)
    def wrapped(data, **kwargs):
        interp_kwargs = kwargs.pop('interp_kwargs', {})
        return f(fill_missing(data, **interp_kwargs), **kwargs)

    return wrapped


@funnel
def pandas_stack(data, names=None, keys=None, verify_integrity=False, sort=False, copy=True, ignore_index=False,
                 levels=None):
    """
    Take a list of DataFrames with the same number of columns and (optionally)
    a list of names (of the same length as the original list; default:
    range(len(x))).  Return a single MultiIndex DataFrame where the original
    DataFrames are stacked vertically, with the data names as their level 1
    indices and their original indices as their level 2 indices.

    INPUTS
    data: data in any format supported by datawrangler

    Also takes all keyword arguments from pandas.concat except axis, join, join_axes

    All other keyword arguments (if any) are passed to funnel

    OUTPUTS
    a single MultiIndex DataFrame
    """

    if is_multiindex_dataframe(data):
        return data
    elif is_dataframe(data):
        data = [data]
    elif len(data) == 0:
        return None

    assert len(np.unique([d.shape[1] for d in data])) == 1, 'All DataFrames must have the same number of columns'
    for i, d1 in enumerate(data):
        template = d1.columns.values
        for d2 in data[(i + 1):]:
            assert np.all([(c in template) for c in d2.columns.values]), 'All DataFrames must have the same columns'

    if keys is None:
        keys = np.arange(len(data), dtype=int)

    assert is_array(keys) or (type(keys) == list), f'keys must be None or a list or array of length len(data)'
    assert len(keys) == len(data), f'keys must be None or a list or array of length len(data)'

    if names is None:
        names = ['ID', *[f'ID{i}' for i in range(1, len(data[0].index.names))], None]

    return pd.concat(data, axis=0, join='outer', names=names, keys=keys,
                     verify_integrity=verify_integrity, sort=sort, copy=copy,
                     ignore_index=ignore_index, levels=levels)


def pandas_unstack(x):
    if not is_multiindex_dataframe(x):
        if is_dataframe(x):
            return x
        else:
            raise Exception(f'Unsupported datatype: {type(x)}')

    names = list(x.index.names)
    grouper = 'ID'
    if not (grouper in names):
        names[0] = grouper
    elif not (names[0] == grouper):
        for i in np.arange(
                len(names)):    # trying n things other than 'ID'; one must be outside of the n-1 remaining names
            next_grouper = f'{grouper}{i}'
            if not (next_grouper in names):
                names[0] = next_grouper
                grouper = next_grouper
                break
    assert names[0] == grouper, 'Unstacking error'

    x.index.rename(names, inplace=True)
    return [d[1].set_index(d[1].index.get_level_values(1)) for d in list(x.groupby(grouper))]


def apply_stacked(f):

    @funnel
    def wrapped(data, *args, **kwargs):
        stack_result = is_multiindex_dataframe(data)

        stacked_data = pandas_stack(data)
        transformed = f(stacked_data, *args, **kwargs)

        if (not stack_result) and (('return_model' in kwargs.keys()) and kwargs['return_model']):
            transformed[0] = pandas_unstack(transformed[0])
        return transformed

    return wrapped


def apply_unstacked(f):

    @funnel
    def wrapped(data, *args, **kwargs):
        stack_result = is_multiindex_dataframe(data)

        unstacked_data = pandas_unstack(data)
        # noinspection PyArgumentList
        transformed = list_generalizer(f)(unstacked_data, *args, **kwargs)

        return_model = kwargs.pop('return_model', False)
        if return_model:
            # noinspection PyTypeChecker
            model = [t[1] for t in transformed]
            # noinspection PyTypeChecker
            transformed = [t[0] for t in transformed]

        if stack_result:
            transformed = pandas_stack(transformed)

        if return_model:
            # noinspection PyUnboundLocalVariable
            return transformed, model
        else:
            return transformed

    return wrapped
