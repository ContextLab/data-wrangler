# Data Wrangler 🤠
Wrangle your messy data into consistent well-organized formats!

### Development status

![pytest](https://github.com/ContextLab/data-wrangler/actions/workflows/ci.yaml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/data-wrangler/badge/?version=latest)](https://data-wrangler.readthedocs.io/en/latest/?badge=latest)


## Overview

Datasets come in all shapes and sizes, and are often *messy* to work with:
  - Observations come in different formats
  - There are missing values
  - It's frustrating to write code to handle all potential use cases
  - Datasets need to be [wrangled](https://en.wikipedia.org/wiki/Data_wrangling) into formats that play more nicely with other parts of your analysis pipeline

The main goals of `data-wrangler` are to:
  1. Help turn messy data into clean(er) data.
  2. Make it easier to adapt existing code (or write new code) without worrying about data formatting details
 
## Supported data formats 

This package, in its current form, is aimed primarily at two broad categories of data:
  - *Numerical data* (e.g., a timeseries of one or more measurements, tables of counts, images, etc.)
  - *Text data* (e.g., words and/or documents)

Specifically, any of the following datatypes can be "wrangled" automatically:
  - `array`-like or `list`-like objects with numerical entries, including high-dimensional arrays or tensors
  - `DataFrame`-like or `Series`-like objects with numerical entries
  - text data (text is embedded using cutting-edge [natural language processing](https://en.wikipedia.org/wiki/Word_embedding) models)
or lists of mixtures of the above.
  - images
  - files from a wide range of formats that contain data in any of the above formats
  - mixed lists of any of the above

Missing observations (e.g. `nan`s, empty strings, etc.) may be filled in using:
  - [Imputation](https://en.wikipedia.org/wiki/Imputation_(statistics)) to fill isolated missing data (e.g., a few columns from a few rows in a data table)
  - [Interpolation](https://en.wikipedia.org/wiki/Interpolation) to fill in missing rows (e.g., when every column within one or more rows is/are missing)


## Installation

To install the `data-wrangler`, run:
```
pip install git+https://github.com/ContextLab/data-wrangler.git
```

# Wrangling your data with `data-wrangler`

## The `wrangle` function

For numerical data (stored in `array`-like, `DataFrame`-like, or `Series`-like objects, or lists thereof), the `wrangle` function will (by default) transform the dataset as follows:
  - Convert `list`-like, `array`-like and `Series`-like objects into `DataFrame`s
  - If the data are passed in as a list, return the resulting list of converted `DataFrame`s
  - If the data are passed as a single `array`-like, `DataFrame`-like, `Series`-like objct, return the resulting converted `DataFrame`
  - Text data may be transformed using any [scikit-learn](https://scikit-learn.org/stable/) or [hugging-face](https://huggingface.co/) model; `data-wrangler` provides wrappers for easily training and/or applying natural language processing models, downloading corpora, etc.

Using the function in this way is simple:

```python
import datawrangler as dw

data = <LOAD IN YOUR MIXED-FORMATTED DATASET>

wrangled_data = dw.wrangle(data) # wrangled_data is either a DataFrame or a list of DataFrames
```

### Customizing how data get (automatically) wrangled

# Decorators
## The `funnel` decorator

## `list_generalizer`

## `interpolate`

## `apply_stacked` and `apply_unstacked`

# Other useful wrangling functions

## Manipulating and evaluating objects

### `update_dict`

### `btwn`

### `dataframe_like`

### `array_like`

### `depth`

## Wrangling (locally and/or remotely) stored data

### `load`

### `save`

# Handling text data

## `scikit-learn` models

## `hugging-face` models

## Downloading supported text corpora

# Adding support for new datatypes

## Writing a `wrangle_<DATATYPE>` function

## Writing an `is_<DATATYPE>` function

# Customizing defaults using `config.ini`

# Contributing

