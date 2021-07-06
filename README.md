# Data Wrangler 🤠
Wrangle your messy data into consistent well-organized formats!


## Overview

Datasets come in all shapes and sizes, and are often *messy*:
  - Observations come in different formats
  - There are missing values
  - Labels are missing and/or aren't consistent
  - Datasets need to be [wrangled](https://en.wikipedia.org/wiki/Data_wrangling) into formats that play more nicely with other parts of your analysis pipeline

The main goal of `data-wrangler` is to turn messy data into clean(er) data, defined as one of the following:
  - A single two-dimensional `numpy` `array` whose rows are observations and whose columns are features
  - A single `pandas` `DataFrame` whose indices are observations and whose columns are features
  - A list of `array`s (each formatted as described above)
  - A list of `DataFrame`s (each formatted as described above)
 
## Supported data formats 

One package can't accomodate every foreseeable format or input source, but `data-wrangler` provides a framework for adding support for new datatypes in a straightforward way.  Essentially, adding support for a new data type entails writing two functions:
  - An `is_<datatype>` function, which should return `True` if an object is compatable with the given datatype (or format), and `False` otherwise
  - A `wrangle_<datatype>` function, which should take in an object of the given type or format and return a `pandas` `DataFrame` with numerical entries

Currently supported datatypes are limited to:
  - `array`-like objects
  - `DataFrame`-like or `Series`-like objects
  - text data (text is embedded using cutting-edge [natural language processing](https://en.wikipedia.org/wiki/Word_embedding) models)
or lists of mixtures of the above.

Missing observations (e.g. `nan`s, empty strings, etc.) may be filled in using:
  - [Probabilistic principle component analysis](https://www.jstor.org/stable/2680726) to fill in isolated missing columns (features) that are present in at least some observations
  - [Interpolation](https://en.wikipedia.org/wiki/Interpolation) to fill in observations with no features (e.g., when nearby observations are available)


## Installation

The `data-wrangler` package may be installed using `pip`:
```
pip install data-wrangler
```

To install the bleeding edge (development) version, use:
```
pip install git+https://github.com/ContextLab/data-wrangler.git
```

## Use

For numerical data (stored in `array`-like, `DataFrame`-like, or `Series`-like objects, or lists thereof), the `wrangle` function will (by default) transform the dataset as follows:
  - Convert `array`-like and `Series`-like objects into `DataFrame`s
  - If the data are passed in as a list, return the resulting list of converted `DataFrame`s
  - If the data are passed as a single `array`-like, `DataFrame`-like, `Series`-like objct, return the resulting converted `DataFrame`

Using the function in this way is simple:

```python
import datawrangler as dw

data = <LOAD IN YOUR MIXED-FORMATTED DATASET>

wrangled_data = dw.wrangle(data) # wrangled_data is either a DataFrame or a list of DataFrames
```


