=============================
Migration Guide: v0.2 → v0.3
=============================

This guide helps you migrate from data-wrangler v0.2.x to v0.3.0, which includes significant modernization and breaking changes.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview of Changes
===================

Version 0.3.0 represents a major modernization of data-wrangler with focus on:

- **Modern Python Support**: Requires Python 3.9+ (dropped support for 3.6-3.8)
- **NumPy 2.0+ Compatibility**: Full support for latest NumPy versions
- **Pandas 2.0+ Compatibility**: Updated for modern pandas API
- **Text Processing Overhaul**: Migrated from Flair to sentence-transformers
- **Dependency Cleanup**: Removed conflicting and deprecated libraries

Breaking Changes
================

Python Version Requirements
----------------------------

**Before (v0.2.x)**::

    # Supported Python 3.6, 3.7, 3.8, 3.9, 3.10
    python_requires=">=3.6"

**After (v0.3.0)**::

    # Requires Python 3.9+
    python_requires=">=3.9"

**Migration**: Upgrade to Python 3.9 or later before installing v0.3.0.

Text Embedding Models
---------------------

The most significant change is the migration from Flair to sentence-transformers for text embeddings.

**Before (v0.2.x)**::

    # Old Flair syntax
    flair_model = {
        'model': 'TransformerDocumentEmbeddings', 
        'args': ['bert-base-uncased']
    }
    
    embeddings = dw.wrangle(texts, text_kwargs={'model': flair_model})

**After (v0.3.0)**::

    # New sentence-transformers syntax
    sentence_model = {
        'model': 'all-mpnet-base-v2',
        'args': [],
        'kwargs': {}
    }
    
    # Or simplified:
    embeddings = dw.wrangle(texts, text_kwargs={'model': 'all-mpnet-base-v2'})

Model Name Mappings
~~~~~~~~~~~~~~~~~~~

Here are recommended migrations for common Flair models:

.. list-table:: Flair → Sentence-Transformers Migration
   :header-rows: 1
   :widths: 50 50

   * - Old Flair Model
     - New Sentence-Transformers Model
   * - ``TransformerDocumentEmbeddings('bert-base-uncased')``
     - ``'all-MiniLM-L6-v2'`` (fast) or ``'all-mpnet-base-v2'`` (quality)
   * - ``TransformerDocumentEmbeddings('roberta-base')``
     - ``'all-distilroberta-v1'``
   * - Custom transformer models
     - Use model name directly from HuggingFace Hub

Installation Changes
--------------------

**Before (v0.2.x)**::

    pip install data-wrangler

**After (v0.3.0)**::

    # Basic installation (sklearn text processing only)
    pip install pydata-wrangler
    
    # Full installation with sentence-transformers support
    pip install "pydata-wrangler[hf]"

Note: The package name on PyPI is now ``pydata-wrangler`` to avoid conflicts.

Removed Dependencies
====================

The following dependencies were removed in v0.3.0:

- ``flair`` - Replaced with sentence-transformers
- ``gensim`` - Caused NumPy version conflicts
- ``konoha`` - Unused Japanese tokenizer
- ``pytorch-transformers`` - Renamed to ``transformers``
- ``pytorch-pretrained-bert`` - Replaced by ``transformers``

If your code relied on these libraries directly, you'll need to install them separately.

Step-by-Step Migration
======================

1. Check Python Version
------------------------

Ensure you're using Python 3.9 or later::

    python --version
    # Should show 3.9.x or higher

2. Update Installation
----------------------

Uninstall old version and install new::

    pip uninstall data-wrangler
    pip install "pydata-wrangler[hf]"

3. Update Text Processing Code
------------------------------

Replace Flair model specifications::

    # OLD - Replace this
    old_model = {'model': 'TransformerDocumentEmbeddings', 'args': ['bert-base-uncased']}
    
    # NEW - With this
    new_model = 'all-mpnet-base-v2'  # or {'model': 'all-mpnet-base-v2', 'args': [], 'kwargs': {}}

4. Test Your Code
-----------------

Run your existing code to identify any remaining issues::

    python -m pytest tests/  # If you have tests
    python your_script.py     # Test your main scripts

Common Migration Issues
=======================

Issue: Import Errors
---------------------

**Problem**::

    ImportError: No module named 'flair'

**Solution**: Remove any direct Flair imports and use data-wrangler's text processing instead::

    # Remove this
    from flair.embeddings import TransformerDocumentEmbeddings
    
    # Use this instead
    import datawrangler as dw
    embeddings = dw.wrangle(texts, text_kwargs={'model': 'all-mpnet-base-v2'})

Issue: Model Configuration Errors
----------------------------------

**Problem**::

    ValueError: Model 'TransformerDocumentEmbeddings' not found

**Solution**: Update model specifications to use sentence-transformers model names.

Issue: Performance Differences
-------------------------------

Sentence-transformers models may have different performance characteristics than Flair models:

- **Speed**: sentence-transformers is generally faster
- **Memory**: Model memory usage may differ  
- **Accuracy**: Results may vary slightly due to different model architectures

Test your specific use case and adjust model choices if needed.

Recommended Model Choices
=========================

For Different Use Cases
------------------------

.. list-table:: Model Recommendations
   :header-rows: 1
   :widths: 30 35 35

   * - Use Case
     - Fast Option
     - High Quality Option
   * - General text similarity
     - ``all-MiniLM-L6-v2``
     - ``all-mpnet-base-v2``
   * - Semantic search
     - ``all-MiniLM-L6-v2``
     - ``all-mpnet-base-v2``
   * - Paraphrase detection
     - ``paraphrase-MiniLM-L6-v2``
     - ``paraphrase-mpnet-base-v2``
   * - Multi-language
     - ``paraphrase-multilingual-MiniLM-L12-v2``
     - ``paraphrase-multilingual-mpnet-base-v2``

Performance Comparison
----------------------

Approximate performance characteristics:

- **all-MiniLM-L6-v2**: 384 dimensions, ~120MB, fastest
- **all-mpnet-base-v2**: 768 dimensions, ~420MB, highest quality
- **paraphrase-MiniLM-L6-v2**: 384 dimensions, optimized for similarity

Testing Your Migration
=======================

Validation Checklist
---------------------

After migration, verify:

☐ Python version is 3.9+
☐ Installation successful: ``pip show pydata-wrangler``
☐ Basic functionality: ``import datawrangler as dw; dw.wrangle([1,2,3])``
☐ Text processing: ``dw.wrangle(["test"], text_kwargs={'model': 'all-MiniLM-L6-v2'})``
☐ Your specific use cases still work correctly
☐ Performance is acceptable for your needs
☐ Results are consistent with your expectations

Sample Test Script
------------------

Use this script to validate your migration::

    import datawrangler as dw
    import numpy as np
    
    # Test basic functionality
    print("Testing basic array wrangling...")
    result = dw.wrangle(np.random.randn(5, 3))
    print(f"Array result shape: {result.shape}")
    
    # Test text processing
    print("\\nTesting text processing...")
    texts = ["Hello world", "Data science is great"]
    text_result = dw.wrangle(texts, text_kwargs={'model': 'all-MiniLM-L6-v2'})
    print(f"Text result shape: {text_result.shape}")
    
    # Test decorator functionality
    print("\\nTesting @funnel decorator...")
    @dw.funnel
    def compute_mean(data):
        return data.mean().mean()
    
    mean_result = compute_mean([1, 2, 3, 4, 5])
    print(f"Mean result: {mean_result}")
    
    print("\\nMigration validation complete!")

Getting Help
============

If you encounter issues during migration:

1. **Check the documentation**: Updated examples in tutorials
2. **Review error messages**: Often contain specific guidance
3. **Test with simple examples**: Isolate the problem
4. **Compare v0.2 vs v0.3 behavior**: Use the examples above

For additional support:

- **GitHub Issues**: https://github.com/ContextLab/data-wrangler/issues
- **Documentation**: https://data-wrangler.readthedocs.io/
- **Examples**: See the tutorials for v0.3.0 patterns

Benefits of v0.3.0
===================

While migration requires some work, v0.3.0 provides significant benefits:

✅ **Better Performance**: Modern dependencies and optimizations
✅ **Future-Proof**: Compatible with latest Python ecosystem
✅ **Improved Models**: Access to state-of-the-art sentence-transformers
✅ **Cleaner Dependencies**: Removed conflicts and deprecated packages
✅ **Better Maintenance**: Built on actively maintained libraries
✅ **Enhanced Documentation**: Comprehensive tutorials and examples

The migration effort pays off with a more robust, performant, and maintainable codebase.