"""Top-level package for datawrangler."""

__author__ = """Contextual Dynamics Lab"""
__email__ = 'contextualdynamics@gmail.com'

from .zoo import wrangle
from .decorate.decorate import funnel, pandas_stack as stack, pandas_unstack as unstack
from .core.configurator import __version__