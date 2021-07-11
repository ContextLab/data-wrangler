"""Top-level package for datawrangler."""

__author__ = """Contextual Dynamics Lab"""
__email__ = 'contextualdynamics@gmail.com'
__version__ = '0.1.0'

from .format import wrangle
from .helpers import btwn, dataframe_like, array_like, load_dataframe
from .io import load, save
from .decorate import list_generalizer, funnel, fill_missing, interpolate, stack_handler, module_checker, \
    unstack_apply, stack_apply, apply_defaults

# TODO: organize imports into modules (need to make folders with their own __init__.py files)