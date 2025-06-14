"""
Lazy import utilities for data-wrangler.

This module provides infrastructure for lazy loading of heavy dependencies,
significantly reducing import time when those dependencies aren't needed.
"""

import importlib
import sys
from functools import wraps


class LazyModule:
    """A placeholder for a module that hasn't been imported yet."""
    
    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None
    
    def _load(self):
        """Load the actual module."""
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module
    
    def __getattr__(self, name):
        """Load module and get attribute when accessed."""
        module = self._load()
        return getattr(module, name)
    
    def __dir__(self):
        """Load module and return directory."""
        module = self._load()
        return dir(module)


def lazy_import(module_name, attribute=None):
    """
    Create a lazy import function.
    
    Parameters
    ----------
    module_name : str
        The name of the module to import
    attribute : str, optional
        Specific attribute to import from the module
        
    Returns
    -------
    function
        A function that imports and returns the module/attribute when called
    """
    def _import():
        module = importlib.import_module(module_name)
        if attribute:
            return getattr(module, attribute)
        return module
    
    # Cache the result after first import
    _import._cached = None
    
    def _cached_import():
        if _import._cached is None:
            _import._cached = _import()
        return _import._cached
    
    return _cached_import


def lazy_import_with_fallback(module_name, attribute=None, fallback_message=None):
    """
    Create a lazy import function with error handling.
    
    Parameters
    ----------
    module_name : str
        The name of the module to import
    attribute : str, optional
        Specific attribute to import from the module
    fallback_message : str, optional
        Custom error message if import fails
        
    Returns
    -------
    function
        A function that imports and returns the module/attribute when called,
        or raises ImportError with custom message
    """
    def _import():
        try:
            module = importlib.import_module(module_name)
            if attribute:
                return getattr(module, attribute)
            return module
        except ImportError as e:
            if fallback_message:
                raise ImportError(fallback_message) from e
            raise
    
    # Cache the result after first import
    _import._cached = None
    
    def _cached_import():
        if _import._cached is None:
            _import._cached = _import()
        return _import._cached
    
    return _cached_import


def requires_import(*modules):
    """
    Decorator that ensures modules are available before function execution.
    
    Parameters
    ----------
    *modules : str
        Module names that must be importable
        
    Returns
    -------
    decorator
        A decorator that checks module availability
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for module in modules:
                try:
                    importlib.import_module(module)
                except ImportError:
                    raise ImportError(
                        f"{module} is required for {func.__name__}. "
                        f"Install with: pip install 'pydata-wrangler[hf]'"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Pre-defined lazy importers for common heavy dependencies
get_sklearn = lazy_import('sklearn')
get_numpy = lazy_import('numpy')
get_pandas = lazy_import('pandas')
get_polars = lazy_import('polars')  # Now a required dependency
get_torch = lazy_import_with_fallback(
    'torch',
    fallback_message="PyTorch not installed. Install with: pip install torch"
)
get_transformers = lazy_import_with_fallback(
    'transformers',
    fallback_message="Transformers not installed. Install with: pip install 'pydata-wrangler[hf]'"
)
get_sentence_transformers = lazy_import_with_fallback(
    'sentence_transformers',
    fallback_message="sentence-transformers not installed. Install with: pip install 'pydata-wrangler[hf]'"
)
get_datasets = lazy_import_with_fallback(
    'datasets',
    fallback_message="datasets not installed. Install with: pip install 'pydata-wrangler[hf]'"
)

# Lazy importers for sklearn submodules
get_sklearn_decomposition = lazy_import('sklearn.decomposition')
get_sklearn_manifold = lazy_import('sklearn.manifold')
get_sklearn_mixture = lazy_import('sklearn.mixture')
get_sklearn_feature_extraction_text = lazy_import('sklearn.feature_extraction.text')
get_sklearn_impute = lazy_import('sklearn.impute')