from configparser import ConfigParser
from pkg_resources import get_distribution
from copy import copy
import os
import warnings
import functools
from flair import embeddings           # used when applying default options
import numpy as np                     # used when applying default options


__version__ = get_distribution('datawrangler')


def get_default_options(fname=None):
    """
    Parse a config.ini file

    Parameters
    ----------
    fname: absolute-path filename for the config.ini file (default: data-wrangler/datawrangler/core/config.ini)

    Returns
    -------
    A dictionary whose keys are function names and whose values are dictionaries of default arguments and keyword
    arguments
    """
    if fname is None:
        fname = os.path.join(os.path.dirname(__file__), 'config.ini')

    config = ConfigParser()
    config.read(fname)
    config = dict(config)

    for a, b in config.items():
        config[a] = dict(b)
        for c, d in config[a].items():
            config[a][c] = eval(d)

    return config


def update_dict(template, updates):
    """
    Replace a template dictionary's values with new values defined in a second "updates" dictionary.

    Parameters
    ----------
    template: default keys and values to use (if not specified in the "updates" dictionary)
    updates: new values to use (and/or new keys to add to the resulting dictionary)

    Returns
    -------
    A new dictionary containing the union of the keys/values in template and updates, with preference given to
    the updates dictionary
    """
    template = copy(template)
    for k, v in updates.items():
        template[k] = v
    return template


defaults = get_default_options()

if not os.path.exists(defaults['data']['datadir']):
    os.makedirs(defaults['data']['datadir'])


# add in default keyword arguments (and values) specified in config.ini based on the function or class name
# can also be used as a decorator
def apply_defaults(f):
    """
    Replace a function's default arguments and keyword arguments with defaults specified in config.ini

    Parameters
    ----------
    f: a function

    Returns
    -------
    a function replacing and un-specified arguments with the defaults defined in config.ini
    """
    def get_name(func):
        if hasattr(func, '__name__'):
            return func.__name__
        else:
            # noinspection PyShadowingNames
            name = str(func)
            if '(' in name:
                return name[:name.rfind('(')]
            else:
                return name

    name = get_name(f)
    if name in defaults.keys():
        default_kwargs = {k: v for k, v in dict(defaults[name]).items() if k[:2] != '__'}
    else:
        default_kwargs = {}

    default_args = [v for k, v in dict(defaults[name]).items() if k[:2] == '__']

    @functools.wraps(f)
    def wrapped_function(*args, **kwargs):
        if len(args) > 0:
            return f(*args, **update_dict(default_kwargs, kwargs))
        else:
            return f(*default_args, **update_dict(default_kwargs, kwargs))
    
    if callable(f):
        return wrapped_function
    else:
        warnings.warn('class decoration is under development and should not be used in critical applications')

        class WrappedClass(f):
            def __init__(self, *args, **kwargs):
                kwargs = update_dict(default_kwargs, kwargs)
                super().__init__(self, *args, **kwargs)

                for a in functools.WRAPPER_ASSIGNMENTS:
                    setattr(self, a, getattr(f, a))

            def __repr__(self):
                return repr(self.__wrapped__)

        return WrappedClass
