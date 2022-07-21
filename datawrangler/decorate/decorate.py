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
format_checkers = eval(defaults['supported_formats']['types'])


# import all model-like classes within a sklearn-like module; return a list of model names
# note: this is NOT a decorator-- it's a helper function used to seed functions for the module_checker decorator
def import_sklearn_models(module):
    """
    Given a Python module object, import all of the scikit-learn-style models it contains (i.e., all objects with
    fit_transform methods) into the workspace

    Parameters
    ----------
    :param module: a Python module object (e.g. sklearn.decomposition)

    Returns
    -------
    :return: a list of valid models contained in the module
    """
    models = [d for d in dir(module) if hasattr(getattr(module, d), 'fit_transform')]
    for m in models:
        exec(f'from {module.__name__} import {m}', globals())
    return models


def get_sklearn_model(x):
    """
    Wrangle a scikit-learn model into a consistent format

    Parameters
    ----------
    :param x: a callable scikit-learn model, a string containing a scikit-learn model's name (e.g.,
    'LatentDirichletAllocation'), or a dictionary with the following keys:
      - 'model': a callable scikit-learn model or a string containing a scikit-learn model's name
      - 'args': a list of arguments to pass to the model (this list will be pre-pended with the data the model is
        applied to)
      - 'kwargs': a list of keyword arguments to pass to the model

    Returns
    -------
    :return: A callable scikit-learn model
    """
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
    """
    Apply one or more scikit-learn models to a dataset

    Parameters
    ----------
    :param model: a scikit-learn model (as defined in the *get_sklearn_model* description), or a list of models to be applied
           in sequence
    :param data: a dataset (array or DataFrame)
    :param args: other arguments to pass to the model (after data)
    :param mode: one of 'fit', 'transform', or 'fit_transform' (default: 'fit_transform'); uses scikit-learn syntax
    :param return_model: if True, both the (potentially transformed) data *and* the fitted model (or list of fitted models)
          are returned.  If False, only the (potentially transformed) data is returned.  Default: False
    :param kwargs: other keyword arguments to pass to *all* of the scikit-learn models (in addition to any model-specific
          keyword arguments)

    Returns
    -------
    :return: Either the (potentially transformed) dataset (if return_model == False) or a tuple containing the
    transformed dataset (first element) and the fitted model(s) (second element), if return_model == True.
    """
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


def list_generalizer(f):
    """
    A decorator that makes a function work for either a single object or a list of objects by calling the function on
    each element

    Parameters
    ----------
    :param f: the function to decorate, of the form f(data, *args, **kwargs).

    Returns
    -------
    :return: A decorated function that supports lists of data objects (rather than only non-list data objects)
    """
    @functools.wraps(f)
    def wrapped(data, *args, **kwargs):
        if type(data) == list:
            return [f(d, *args, **kwargs) for d in data]
        else:
            return f(data, *args, **kwargs)

    return wrapped


def funnel(f):
    """
    A decorator that coerces any data passed into the function into a pandas DataFrame or a list of DataFrames

    Parameters
    ----------
    :param f: a function of the form f(data, *args, **kwargs) that assumes data is either a DataFrame or a list of
       DataFrames

    Returns
    -------
    :return: A decorated function the supports any wrangle-able data format
    """
    @functools.wraps(f)
    def wrapped(data, *args, **kwargs):
        wrangle_kwargs = kwargs.pop('wrangle_kwargs', {})
        for fc in format_checkers:
            wrangle_kwargs[f'{fc}_kwargs'] = kwargs.pop(f'{fc}_kwargs', {})

        return f(wrangle(data, **wrangle_kwargs), *args, **kwargs)

    return wrapped


def interpolate(f):
    """
    A decorator that fills in missing data by imputing and/or interpolating missing values

    Parameters
    ----------
    :param f: a function of the form f(data, *args, **kwargs) that assumes the data are formatted as either a DataFrame or
       a list of DataFrames, with no missing (numpy.nan) values

    Returns
    -------
    :return: A decorated function that supports any wrangle-able datatype.  Pass in the following keyword arguments to
    fill in missing data:
      impute_kwargs: a dictionary containing one or more scikit-learn imputation models (e.g.,
          {'model': 'IterativeImputer'}.  The 'model' can be specified as defined in the *apply_sklearn_model* function.
      any other keywords are passed to pandas.DataFrame.interpolate; e.g. method='linear' will apply linear
          interpolation to fill in missing values.  A full list of supported arguments may be found here:
          https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
          If no other keyword arguments are specified, no interpolation is performed.
    """
    @funnel
    def fill_missing(data, return_model=False, **kwargs):
        impute_kwargs = kwargs.pop('impute_kwargs', {})

        if impute_kwargs:
            model = impute_kwargs.pop('model', eval(defaults['impute']['model']))
            imputed_data, model = apply_sklearn_model(model, data, return_model=True, **impute_kwargs)
            data = pd.DataFrame(data=imputed_data, index=data.index, columns=data.columns)
        else:
            model = None

        if kwargs:
            kwargs = update_dict(defaults['interpolate'], kwargs, from_config=True)
            data = data.interpolate(**kwargs)

        if return_model:
            return data, {'model': model, 'args': [], 'kwargs': kwargs}
        else:
            return data

    @functools.wraps(f)
    def wrapped(data, *args, **kwargs):
        interp_kwargs = kwargs.pop('interp_kwargs', {})
        return f(fill_missing(data, *args, **interp_kwargs), **kwargs)

    return wrapped


# noinspection PyIncorrectDocstring
def pandas_stack(data, names=None, keys=None, verify_integrity=False, sort=False, copy=True, ignore_index=False,
                 levels=None):
    """
    Take a list of DataFrames with the same number of columns and (optionally)
    a list of names (of the same length as the original list; default:
    range(len(x))).  Return a single MultiIndex DataFrame where the original
    DataFrames are stacked vertically, with the data names as their level 1
    indices and their original indices as their level 2 indices.

    Parameters
    ----------
    :param data: A single DataFrame or a list of DataFrames with must matching columns.
    :param names: names for the levels in the resulting hierarchical index. (Default: None)
    :param keys: if multiple levels passed, should contain tuples. Construct hierarchical index using the passed keys as
      the outermost level.
    :param verify_integrity: check whether the new concatenated axis contains duplicates. This can be very expensive
      relative to the actual data concatenation.  (Default: False)
    :param sort: sort non-concatenation axis if it is not already aligned when join is ‘outer’. This has no effect when
      join='inner', which already preserves the order of the non-concatenation axis.  (Default: False)
    :param copy: if False, do not copy data unnecessarily.  (Default: True)
    :param ignore_index: if True, do not use the index values along the concatenation axis. The resulting axis will be
      labeled 0, …, n - 1. This is useful if you are concatenating objects where the concatenation axis does not have
      meaningful indexing information. Note the index values on the other axes are still respected in the join.
      (Default: False)
    :param levels: specific levels (unique values) to use for constructing a MultiIndex. Otherwise they will be inferred
      from the keys. (Default: None)
    :param kwargs: any other keyword arguments will be passed to datawrangler.decorate.funnel

    Returns
    -------
    :return: a single MultiIndex DataFrame
    """

    if is_multiindex_dataframe(data):
        return data
    elif is_dataframe(data):
        data = [data]
    elif len(data) == 0:
        return None

    # ensure that Series objects are cast into DataFrames
    data = [pd.DataFrame(d) for d in data]

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
    """
    Turn a MultiIndex DataFrame into a list of DataFrames, using the unique top-level index values to divide the data.
    If the dataset is a "regular" (non-MultiIndex) DataFrame, it is returned as a list with a single element,
    containing the un-modified DataFrame

    Parameters
    ----------
    :param x: a single DataFrame, a MultiIndex DataFrame, or a list of DataFrames

    Returns
    -------
    :return: a list of one or more DataFrames
    """
    if not is_multiindex_dataframe(x):
        if is_dataframe(x):
            return x
        elif issubclass(type(x), pd.Series):
            return pd.DataFrame(x).T
        elif type(x) is list and all([is_dataframe(d) for d in x]):
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

    groups = list(x.groupby(grouper))
    n_levels = len(groups[0][1].index.levels)
    if n_levels > 2:
        g = groups[0][1]
        index = pd.MultiIndex.from_arrays([g.index.get_level_values(len(g.index.levels) - n)
                                           for n in range(1, len(g.index.levels))][::-1])
        return [d[1].set_index(index) for d in groups]
    else:
        return [d[1].set_index(d[1].index.get_level_values(len(d[1].index.levels) - 1))
                for d in list(x.groupby(grouper))]


def apply_stacked(f):
    """
    Decorate a function to adjust how it handles data as follows:
      - Wrangle the data into DataFrames (the resulting DataFrames must all have the same number of columns).
        MultiIndex DataFrames are also supported (and can represent already-stacked datasets)
      - Vertically concatenate the wrangled data
      - Apply the function to the "stacked" dataset, treating the combined data as a "single" DataFrame
      - If the original dataset was provided in "unstacked" format, unstack the result into a list of DataFrames
      - Return the resulting (stacked or unstacked) DataFrame(s)

    Parameters
    ----------
    :param f: a function of the form f(data, *args, **kwargs) that assumes data is a single DataFrame, and that returns a
       single DataFrame as output.

    Returns
    -------
    :return: a decorated function that supports any wrangle-able data types, applies the original function to the full
    list of datasets simultaneously, and then returns the result(s) as a new DataFrame or list of DataFrames.
    """

    @funnel
    def wrapped(data, *args, **kwargs):
        def helper(d, x, split):
            if split:
                x0 = x[0]
                x1 = x[1]
            else:
                x0 = x

            if is_dataframe(d) and type(x0) is list and len(x0) == 1:
                x0 = x0[0]

            if split:
                return x0, x1
            else:
                return x0

        stack_result = is_multiindex_dataframe(data)

        stacked_data = pandas_stack(data)
        transformed = f(stacked_data, *args, **kwargs)

        return_model = kwargs.copy().pop('return_model', False)
        if not stack_result:
            if ('return_model' in kwargs.keys()) and kwargs['return_model']:
                transformed[0] = pandas_unstack(transformed[0])
            else:
                transformed = pandas_unstack(transformed)

        return helper(data, transformed, return_model)

    return wrapped


def apply_unstacked(f):
    """
    Decorate a function to adjust how it handles data as follows:
      - Wrangle the data into a list of DataFrames.  MultiIndex DataFrames are also supported (and can represent stacked
        datasets)
      - Apply the function (individually) to each DataFrame in the resulting list
      - If the original dataset was provided in "stacked" format, stack the result into a MultiIndex DataFrame
      - Return the resulting (stacked or unstacked) DataFrame(s)

    Parameters
    ----------
    :param f: a function of the form f(data, *args, **kwargs) that assumes data is a single DataFrame, and that returns
      a single DataFrame as output.

    Returns
    -------
    :return: A decorated function that supports any wrangle-able data types, applies the original function to the full
    list of datasets separately, and then returns the result(s) as a new DataFrame or list of DataFrames.
    """

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
