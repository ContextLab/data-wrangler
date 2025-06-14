"""
Data Wrangler: Transform messy data into clean DataFrames (pandas or Polars)

Data Wrangler is a Python package that automatically transforms various data types
(arrays, text, files, URLs, etc.) into clean, consistent DataFrame format using 
either pandas or Polars backends. It specializes in text data processing using 
modern NLP models and offers 2-100x performance improvements with Polars.

Key Features:
- Dual backend support: pandas (default) or Polars for high performance
- Automatic data type detection and conversion
- Text embedding using sentence-transformers and sklearn models
- Function decorators for seamless DataFrame integration
- Support for files, URLs, and mixed data types
- Configurable processing pipeline

Basic Usage:
    >>> import datawrangler as dw
    >>> df = dw.wrangle(your_data)  # pandas DataFrame (default)
    >>> df_fast = dw.wrangle(your_data, backend='polars')  # Polars DataFrame
    
    # With text data using sentence-transformers
    >>> text_df = dw.wrangle(["Hello world", "Another text"], 
    ...                      text_kwargs={'model': 'all-MiniLM-L6-v2'},
    ...                      backend='polars')  # 2-100x faster with Polars
    
    # Using the @funnel decorator (backend-agnostic)
    >>> @dw.funnel
    ... def your_function(df):
    ...     return df.mean()

Backend Differences:
- pandas: 
  * Full feature compatibility with named indexes
  * Index names preserved during processing
  * Slower performance on large datasets
  * Extensive ecosystem support
- Polars:
  * High performance (2-100x faster on large datasets) 
  * Position-based indexing only (no named indexes)
  * Index names not preserved during backend conversion
  * Limited interpolation support in decorators
  * Growing ecosystem, may have fewer integrations

Choose pandas for: Small datasets, complex index operations, maximum compatibility
Choose Polars for: Large datasets, performance-critical applications, simple workflows

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
