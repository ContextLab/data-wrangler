"""
Data Wrangler: Transform messy data into clean pandas DataFrames

Data Wrangler is a Python package that automatically transforms various data types
(arrays, text, files, URLs, etc.) into clean, consistent pandas DataFrame format.
It specializes in text data processing using modern NLP models.

Key Features:
- Automatic data type detection and conversion
- Text embedding using sentence-transformers and sklearn models
- Function decorators for seamless DataFrame integration
- Support for files, URLs, and mixed data types
- Configurable processing pipeline

Basic Usage:
    >>> import datawrangler as dw
    >>> df = dw.wrangle(your_data)
    
    # With text data using sentence-transformers
    >>> text_df = dw.wrangle(["Hello world", "Another text"], 
    ...                      text_kwargs={'model': 'all-MiniLM-L6-v2'})
    
    # Using the @funnel decorator
    >>> @dw.funnel
    ... def your_function(df):
    ...     return df.mean()

Requirements:
- Python 3.9+
- Optional: Install with [hf] extras for sentence-transformers support

  pip install "pydata-wrangler[hf]"

Version: 0.3.0+ (NumPy 2.0+ and pandas 2.0+ compatible)
"""

__author__ = """Contextual Dynamics Lab"""
__email__ = 'contextualdynamics@gmail.com'

from .zoo import wrangle
from .decorate.decorate import funnel, pandas_stack as stack, pandas_unstack as unstack
from .core import __version__
