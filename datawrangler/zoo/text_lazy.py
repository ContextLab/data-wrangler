import six
import os
import warnings
import numpy as np

from .array import is_array, wrangle_array
from .dataframe import is_dataframe
from .null import is_null

from ..core.configurator import get_default_options, apply_defaults, update_dict
from ..io import load
from ..io.io import get_extension
from ..util.lazy_imports import (
    lazy_import_with_fallback,
    get_sklearn_feature_extraction_text,
    get_sklearn_decomposition,
    get_sentence_transformers,
    get_transformers,
    get_torch,
    get_datasets
)

# Lazy imports for sklearn modules
_get_sklearn_text = lazy_import_with_fallback('sklearn.feature_extraction', 'text')
_get_sklearn_decomposition = lazy_import_with_fallback('sklearn', 'decomposition')

# Lazy imports for HuggingFace modules
_get_SentenceTransformer = lazy_import_with_fallback(
    'sentence_transformers', 'SentenceTransformer',
    fallback_message="sentence-transformers not installed. Install with: pip install 'pydata-wrangler[hf]'"
)

_get_AutoTokenizer = lazy_import_with_fallback(
    'transformers', 'AutoTokenizer',
    fallback_message="transformers not installed. Install with: pip install 'pydata-wrangler[hf]'"
)

_get_AutoModel = lazy_import_with_fallback(
    'transformers', 'AutoModel',
    fallback_message="transformers not installed. Install with: pip install 'pydata-wrangler[hf]'"
)

_get_torch = lazy_import_with_fallback(
    'torch',
    fallback_message="PyTorch not installed. Install with: pip install torch"
)

_get_load_dataset = lazy_import_with_fallback(
    'datasets', 'load_dataset',
    fallback_message="datasets not installed. Install with: pip install 'pydata-wrangler[hf]'"
)

_get_dataset_config_names = lazy_import_with_fallback(
    'datasets', 'get_dataset_config_names',
    fallback_message="datasets not installed. Install with: pip install 'pydata-wrangler[hf]'"
)

_get_list_datasets = lazy_import_with_fallback(
    'huggingface_hub', 'list_datasets',
    fallback_message=None  # Optional dependency
)

# Global variables
defaults = get_default_options()
preloaded_corpora = {}

# Cache for checking if modules are available without importing
_module_availability_cache = {}

def _is_module_available(module_name):
    """Check if a module is available without importing it."""
    if module_name not in _module_availability_cache:
        try:
            import importlib.util
            spec = importlib.util.find_spec(module_name)
            _module_availability_cache[module_name] = spec is not None
        except (ImportError, ValueError):
            _module_availability_cache[module_name] = False
    return _module_availability_cache[module_name]