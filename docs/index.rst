DataWrangler
======================================

**Transform messy data into clean pandas/Polars DataFrames with intelligent automation**

DataWrangler is a powerful Python package that automatically converts diverse data formats into clean, analysis-ready DataFrames. Whether you're working with arrays, text, images, or mixed data types, DataWrangler intelligently detects formats and applies appropriate transformations – all with a simple, unified API.

🚀 **New**: High-performance Polars backend support for 2-100x faster processing!

Why DataWrangler?
-----------------

**🎯 Intelligent Automation**
  No more manual data preprocessing. DataWrangler automatically detects data types and applies appropriate transformations.

**⚡ High Performance** 
  Choose between pandas (familiar) and Polars (fast) backends. Get dramatic speedups with zero code changes.

**🔧 Unified API**
  One simple function handles arrays, text, images, files, URLs, and mixed data types.

**📊 Research-Ready**
  Built for data science workflows with advanced text processing, embeddings, and ML preprocessing.

**🛡️ Production-Tested**
  Robust error handling, comprehensive testing, and battle-tested in real research environments.

Quick Start Examples
--------------------

**Basic Data Wrangling**

.. code-block:: python

   import datawrangler as dw
   import numpy as np

   # Arrays become DataFrames automatically
   array_data = np.random.rand(1000, 5)
   df = dw.wrangle(array_data)
   print(df.head())

**High-Performance with Polars**

.. code-block:: python

   # Same operation, 2-100x faster with Polars backend
   fast_df = dw.wrangle(array_data, backend='polars')
   
   # Set global backend preference
   from datawrangler.core.configurator import set_dataframe_backend
   set_dataframe_backend('polars')  # All operations now use Polars

**Advanced Text Processing**

.. code-block:: python

   # Text documents become embedding vectors
   documents = [
       "Machine learning transforms data into insights",
       "Data science combines statistics with programming",
       "AI enables automated decision-making systems"
   ]
   
   # Automatic text embeddings with state-of-the-art models
   text_df = dw.wrangle(documents)
   print(f"Embedded {len(documents)} documents into {text_df.shape} DataFrame")
   
   # Use modern transformer models for better quality
   sentence_model = {'model': 'all-mpnet-base-v2'}
   embeddings = dw.wrangle(documents, text_kwargs={'model': sentence_model})

**Mixed Data Types in One Call**

.. code-block:: python

   # Process multiple data types simultaneously
   mixed_data = [
       np.random.rand(500, 10),           # NumPy array
       "path/to/image.jpg",               # Image file
       documents,                         # Text documents
       "https://api.example.com/data.csv" # Remote CSV
   ]
   
   results = dw.wrangle(mixed_data, return_dtype=True)
   dataframes, detected_types = results
   
   for df, dtype in zip(dataframes, detected_types):
       print(f"{dtype}: {df.shape}")

**Function Decoration for Seamless Integration**

.. code-block:: python

   from datawrangler.decorate import funnel
   
   @funnel  # Automatically converts inputs to DataFrames
   def analyze_data(df):
       """Your function works with any data type now!"""
       return df.describe()
   
   # Works with arrays, text, files - anything!
   stats = analyze_data(array_data)      # NumPy array
   text_stats = analyze_data(documents)  # Text documents

Common Use Cases
----------------

**🔬 Research & Academia**
  * Literature analysis and text mining
  * Experimental data processing
  * Multi-modal data integration
  * Reproducible research pipelines

**💼 Business Intelligence**
  * Customer feedback analysis
  * Sales data aggregation
  * Performance monitoring dashboards
  * Cross-platform data integration

**🤖 Machine Learning**
  * Feature engineering automation
  * Text preprocessing for NLP models
  * Multi-source data fusion
  * Model input preparation

**📈 Data Engineering**
  * ETL pipeline simplification
  * Real-time data processing
  * Data lake preprocessing
  * Format standardization

Performance Benefits
--------------------

DataWrangler with Polars backend delivers significant performance improvements:

.. code-block:: python

   import time
   
   # Large dataset example
   large_array = np.random.rand(100000, 50)
   
   # Pandas backend (traditional)
   start = time.time()
   pandas_df = dw.wrangle(large_array, backend='pandas')
   pandas_time = time.time() - start
   
   # Polars backend (high-performance)
   start = time.time()
   polars_df = dw.wrangle(large_array, backend='polars')
   polars_time = time.time() - start
   
   speedup = pandas_time / polars_time
   print(f"Polars is {speedup:.1f}x faster!")
   # Typical result: 50-100x speedup for large arrays

**Real-world performance gains:**

* **Array processing**: 2-100x faster conversion
* **Text embeddings**: 3-10x faster document processing  
* **Aggregations**: 5-50x faster group-by operations
* **Memory usage**: 30-70% reduction for large datasets

Getting Started
---------------

1. **Installation**::

    pip install pydata-wrangler

2. **Optional high-performance dependencies**::

    pip install pydata-wrangler[hf]  # Adds transformers, sentence-transformers

3. **Start wrangling**::

    import datawrangler as dw
    df = dw.wrangle(your_data)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   installation
   readme
   migration_guide

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   tutorials
   api

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
