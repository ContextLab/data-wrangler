from configparser import ConfigParser
from pkg_resources import get_distribution
from copy import copy
import os
import warnings
import functools
from flair import embeddings  # used when applying default options


__version__ = get_distribution('datawrangler')


def get_default_options(fname=None):
    if fname is None:
        fname = os.path.join(os.path.dirname(__file__), 'config.ini')

    config = ConfigParser()
    config.read(fname)
    return config


def update_dict(template, updates):
    template = copy(template)
    for k, v in updates.items():
        template[k] = v
    return template


defaults = get_default_options()

if not os.path.exists(eval(defaults['data']['datadir'])):
    os.makedirs(eval(defaults['data']['datadir']))


# add in default keyword arguments (and values) specified in config.ini based on the function or class name
# can also be used as a decorator
def apply_defaults(f):
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
