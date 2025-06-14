=======
History
=======

0.4.0 (2025-06-14)
------------------

**Major Release: High-Performance Polars Backend + Simplified Text API**

This release introduces first-class Polars support for dramatic performance improvements and dramatically simplifies the text model API:

**🚀 NEW: High-Performance Polars Backend (2-100x faster!):**
* **Dual DataFrame Support**: Choose between pandas (default) or Polars backends
* **Zero Code Changes**: Add ``backend='polars'`` to any operation for instant speedups
* **Comprehensive Coverage**: All data types (arrays, text, files) work with both backends
* **Smart Type Preservation**: DataFrames maintain their type when no backend specified
* **Global Configuration**: Set default backend preference with ``set_dataframe_backend('polars')``
* **Cross-Backend Conversion**: Seamlessly convert between pandas and Polars DataFrames

**Performance Gains with Polars:**
* **Array Processing**: 2-100x faster conversion for large datasets
* **Text Embeddings**: 3-10x faster document processing
* **Memory Efficiency**: 30-70% reduction in memory usage
* **Parallel Processing**: Built-in multi-core optimization

**Text Model API Simplification (80% reduction in verbosity):**
* **Simple String Format**: ``{'model': 'all-MiniLM-L6-v2'}`` now works everywhere
* **Automatic Normalization**: All model formats converted to unified dict internally
* **List Support**: Lists of models work with simplified format (e.g., ``['CountVectorizer', 'NMF']``)
* **Full Backward Compatibility**: All existing verbose syntax continues working

**Google Colab Installation Fix:**
* Removed redundant ``configparser`` from requirements.txt (built-in to Python 3.x)
* Eliminated installation warning popup in Google Colab environments
* Cleaner dependency list and faster installation

**Enhanced Documentation:**
* Updated all examples to use simplified text model API
* Added comprehensive Polars backend examples and tutorials
* Made all documentation backend-agnostic with performance guidance
* Fixed all docstring examples to use public API correctly

**Example of New Polars Backend:**

.. code-block:: python

    import datawrangler as dw
    import numpy as np
    
    # Large dataset example
    large_array = np.random.rand(50000, 20)
    
    # Traditional pandas backend
    pandas_df = dw.wrangle(large_array)  # Default
    
    # High-performance Polars backend (2-100x faster!)
    polars_df = dw.wrangle(large_array, backend='polars')
    
    # Set global preference for all operations
    from datawrangler.core.configurator import set_dataframe_backend
    set_dataframe_backend('polars')  # All operations now use Polars

**Example of Simplified Text API:**

Before (v0.3.0)::

    # Verbose dictionary format required
    text_kwargs = {
        'model': {
            'model': 'all-MiniLM-L6-v2',
            'args': [],
            'kwargs': {}
        }
    }

After (v0.4.0)::

    # Simplified - just pass the model name!
    text_kwargs = {'model': 'all-MiniLM-L6-v2'}
    
    # Works with Polars backend too for 3-10x faster text processing!
    fast_embeddings = dw.wrangle(texts, text_kwargs=text_kwargs, backend='polars')

0.3.0 (2025-06-13)
------------------

**Major Release: NumPy 2.0+ Compatibility & Modern ML Libraries**

This release brings full compatibility with NumPy 2.0+ and pandas 2.0+ while modernizing the text embedding infrastructure:

**Breaking Changes:**
* Replaced Flair with sentence-transformers for text embeddings
* Removed gensim dependency (eliminates NumPy version conflicts) 
* Updated text embedding API to use sentence-transformers models

**New Features:**
* Full NumPy 2.0+ and pandas 2.0+ compatibility
* Modern sentence-transformers integration for text embeddings
* Support for latest scikit-learn, matplotlib, and scipy versions
* Enhanced error handling for missing dependencies

**Bug Fixes:**
* Fixed numpy.str_ deprecation that broke in NumPy 2.0+
* Updated HuggingFace datasets import for API changes
* Fixed sklearn IterativeImputer experimental import compatibility
* Replaced deprecated matplotlib.pyplot.imread

**Documentation:**
* Updated all examples to use sentence-transformers syntax
* Modernized installation instructions and model references
* Comprehensive tutorial updates with new embedding approaches

**Migration Guide:**
Old Flair syntax: `{'model': 'TransformerDocumentEmbeddings', 'args': ['bert-base-uncased']}`
New syntax: `{'model': 'all-mpnet-base-v2', 'args': [], 'kwargs': {}}`

0.2.2 (22-07-25)
-----------------

* Better error handling when hugging-face libraries aren't installed and user asks to embed text using hugging-face models

0.2.1 (22-07-25)
------------------

* Bug fixes when hugging-face libraries aren't installed

0.2.0 (2022-07-25)
------------------

* Adds CUDA (GPU) support for pytorch models
* Streamline package by not installing hugging-face support by default
* Adds Python 3.10 support (and associated tests)
* Relaxes some tests to support a wider range of platforms (mostly this is relevant for GitHub CI)
* Relaxes requirements.txt versioning to improve compatibility with other libraries when installing via pip

0.1.7 (2021-08-09)
------------------

* Updates default behaviors for several models (via config.ini)


0.1.6 (2021-08-09)
------------------

* Another bug fix release (more fixes to datawrangler.unstack)

0.1.5 (2021-08-09)
------------------

* Corrected a bug in datawrangler.unstack

0.1.4 (2021-08-04)
------------------

* Added an option to specify a customized dictionary of default options to the apply_default_options function

0.1.3 (2021-08-04)
------------------

* Fixed some bugs related to stacking and unstacking DataFrames

0.1.2 (2021-08-04)
------------------

* Minor update that corrects URLs of Khan Academy and NeurIPS corpora and corrects some issues with loading npy files

0.1.1 (2021-07-19)
------------------

* Minor update in order to make the package available on pipy.

0.1.0 (2021-07-09)
------------------

* First release on PyPI.
