from configparser import ConfigParser
from pkg_resources import get_distribution

import functools

__version__ = get_distribution('datawrangler')


def get_default_options(fname='config.ini'):
    config = ConfigParser()
    config.read(fname)
    return config


defaults = get_default_options()


# add in default keyword arguments (and values) specified in config.ini based on the function name
# can also be used as a decorator
def apply_defaults(f):
    if f.__name__ in defaults.keys():
        default_args = defaults[f.__name__]
    else:
        default_args = {}

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        for k, v in kwargs:
            default_args[k] = v
        return f(*args, **default_args)

    return wrapped
