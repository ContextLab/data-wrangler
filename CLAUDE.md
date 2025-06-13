# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**data-wrangler** is a Python package that transforms messy data into clean pandas DataFrames, with emphasis on text data and NLP. Published as `pydata-wrangler` on PyPI.

## Development Commands

### Core Development Tasks
- `make test` - Run pytest test suite
- `make lint` - Run flake8 code style checks  
- `make coverage` - Generate test coverage reports
- `make docs` - Build Sphinx documentation
- `pytest tests/wrangler/test_specific.py::test_function` - Run specific test

### Build and Release
- `make clean` - Clean all build artifacts
- `make dist` - Build source and wheel packages
- `make install` - Install package locally

### Testing
- `make test-all` - Run tox across Python 3.6-3.10
- Tests located in `/tests/wrangler/` with sample data in `/tests/resources/`

## Architecture

### Core Package Structure
- `datawrangler/core/` - Configuration management via config.ini
- `datawrangler/decorate/` - Function decorators (especially `@funnel`)
- `datawrangler/io/` - File/URL loading with format auto-detection  
- `datawrangler/util/` - Helper utilities
- `datawrangler/zoo/` - Data type handlers (array, dataframe, text, null)

### Key Patterns
1. **Plugin Architecture**: `zoo/format.py` orchestrates priority-based data type detection using `is_<type>` and `wrangle_<type>` functions
2. **Decorator Pattern**: `@funnel` automatically converts inputs to DataFrames for function compatibility
3. **Configuration-Driven**: Uses `core/config.ini` for ML model defaults and processing options
4. **Extensible I/O**: Supports files, URLs, multiple formats with automatic detection

### Main Entry Points
- `datawrangler.wrangle()` - Primary data transformation API
- `datawrangler.decorate.funnel` - Function decorator
- `datawrangler.io.load()` and `datawrangler.io.save()` - I/O operations

## Dependencies
- **Core**: pandas, numpy, scipy, scikit-learn, matplotlib
- **Optional ML**: PyTorch, HuggingFace transformers, sentence-transformers (in `requirements_hf.txt`)
- **Dev**: pytest, Sphinx, flake8 (in `requirements_dev.txt`)

## Adding New Data Types
Implement two functions in the appropriate `zoo/` module:
- `is_<datatype>(obj)` - Type detection function
- `wrangle_<datatype>(obj, **kwargs)` - Conversion to DataFrame

## Development Environment
Use `conda env create -f dev.yaml` for full ML environment setup.