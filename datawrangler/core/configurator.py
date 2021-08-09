from configparser import ConfigParser
from copy import copy
import os
import warnings
import functools
from flair import embeddings           # used when applying default options
import numpy as np                     # used when applying default options


__version__ = '0.1.6'


def get_default_options(fname=None):
    """
    Parse a config.ini file

    Parameters
    ----------
    :param fname: absolute-path filename for the config.ini file (default: data-wrangler/datawrangler/core/config.ini)

    Returns
    -------
    :return: A dictionary whose keys are function names and whose values are dictionaries of default arguments and keyword
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
            config[a][c] = d

    return config


def update_dict(template, updates, from_config=False):
    """
    Replace a template dictionary's values with new values defined in a second "updates" dictionary.

    Parameters
    ----------
    :param template: default keys and values to use (if not specified in the "updates" dictionary)
    :param updates: new values to use (and/or new keys to add to the resulting dictionary)
    :param from_config: if True, evaluate the keys in the "template" dictionary and set their values to the result.
      Used when loading options from the configuration file.  (Default: False)

    Returns
    -------
    :return: A new dictionary containing the union of the keys/values in template and updates, with preference given to
    the updates dictionary
    """
    template = copy(template)
    if from_config:
        for k, v in template.items():
            template[k] = eval(v)

    for k, v in updates.items():
        template[k] = v
    return template


defaults = get_default_options()

if not os.path.exists(eval(defaults['data']['datadir'])):
    os.makedirs(eval(defaults['data']['datadir']))


# add in default keyword arguments (and values) specified in config.ini based on the function or class name
# can also be used as a decorator
def apply_defaults(f, defaults=None):
    """
    Replace a function's default arguments and keyword arguments with defaults specified in config.ini

    Parameters
    ----------
    :param f: a function
    :param defaults: an optional dictionary of default options (default: get_default_options)

    Returns
    -------
    :return: a function replacing and un-specified arguments with the defaults defined in config.ini
    """

    if defaults is None:
        defaults = get_default_options()

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
        default_kwargs = {k: eval(v) for k, v in dict(defaults[name]).items() if k[:2] != '__'}
    else:
        default_kwargs = {}

    default_args = [eval(v) for k, v in dict(defaults[name]).items() if k[:2] == '__']

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
