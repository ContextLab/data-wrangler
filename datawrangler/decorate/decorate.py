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
from ..core import get_default_options, apply_defaults, update_dict


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


# intercept data passed to a function by stacking (or unstacking) the dataframes, applying the given function,
# and then inverting the stack/unstack operation (unless return_override is True)
def stack_handler(apply_stacked=False, return_override=False):
    # noinspection PyUnusedLocal
    @interpolate
    def format_interp_stack_extract(data, keys=None, **kwargs):
        stacked_data = pandas_stack(data, keys=keys)
        vals = stacked_data.values
        return vals, stacked_data

    def decorator(f):
        @functools.wraps(f)
        def wrapped(data, **kwargs):
            def returner(x, rmodel=None, rreturn_model=False):
                if rreturn_model:
                    return rmodel, x
                else:
                    return x

            if 'keys' not in kwargs.keys():
                kwargs['keys'] = None

            if 'stack' not in kwargs.keys():
                kwargs['stack'] = False

            return_model = (not return_override) and ('return_model' in kwargs.keys()) and kwargs['return_model']
            if not return_model:
                kwargs.pop('return_model', None)

            keys = kwargs.pop('keys', None)
            stack = kwargs.pop('stack', None)

            vals, stacked_data = format_interp_stack_extract(data, keys=keys, **kwargs)
            unstacked_data = pandas_unstack(stacked_data)

            # ignore sklearn warnings...this should be written more responsibly :)
            warnings.simplefilter('ignore')

            if apply_stacked:
                transformed = f(stacked_data, **kwargs)
                if return_override:
                    return transformed

                if return_model:
                    model, transformed = transformed
                else:
                    model = None

                transformed = pd.DataFrame(data=transformed, index=stacked_data.index,
                                           columns=np.arange(transformed.shape[1]))
                if stack:
                    return returner(transformed, rmodel=model, rreturn_model=return_model)
                else:
                    return returner(pandas_unstack(transformed), rmodel=model, rreturn_model=return_model)
            else:
                transformed = f([x.values for x in unstacked_data], **kwargs)
                if return_override:
                    return transformed

                if return_model:
                    model, transformed = transformed
                else:
                    model = None

                if stack:
                    return returner(pd.DataFrame(data=np.vstack(transformed), index=stacked_data.index), rmodel=model,
                                    rreturn_model=return_model)
                else:
                    return returner(
                        [pd.DataFrame(data=v, index=unstacked_data[i].index) for i, v in enumerate(transformed)],
                        rmodel=model, rreturn_model=return_model)

        return wrapped

    return decorator


# ensure that the named algorithm is contained within the list of approved algorithms of the given type
def module_checker(modules=None, alg_list=None):
    if modules is None:
        modules = []
    if alg_list is None:
        alg_list = []

    def decorator(f):
        @functools.wraps(f)
        def wrapped(data, **kwargs):
            if 'algorithm' not in kwargs.keys():
                algorithm = defaults[f.__name__]['algorithm']
            else:
                algorithm = kwargs.pop('algorithm', None)

            if is_text(algorithm):
                # security check to prevent executing arbitrary code
                verified = False
                if len(alg_list) > 0:
                    assert any([algorithm in eval(f'{a}_models') for a in alg_list]), f'Unknown {f.__name__} ' \
                                                                                      f'algorithm: {algorithm}'
                    verified = True
                if not verified:
                    assert algorithm in eval(f'{f.__name__}_models'), f'Unknown {f.__name__} algorithm: {algorithm}'
                algorithm = eval(algorithm)

            # make sure a function from the appropriate module is being passed
            if len(modules) > 0:
                assert any([m in algorithm.__module__ for m in modules]), f'Unknown {f.__name__} ' \
                                                                          f'algorithm: {algorithm.__name__}'

            kwargs['algorithm'] = algorithm
            return f(data, **kwargs)

        return wrapped

    return decorator


# unstack the data, apply the given function, then re-stack if needed
@stack_handler(apply_stacked=False)
def unstack_apply(data, **kwargs):
    assert 'algorithm' in kwargs.keys(), 'must specify algorithm'
    return algorithm(data, **kwargs)


# stack the data, apply the given function, then unstack if needed
@stack_handler(apply_stacked=True)
def stack_apply(data, **kwargs):
    assert 'algorithm' in kwargs.keys(), 'must specify algorithm'
    return algorithm(data, **kwargs)

