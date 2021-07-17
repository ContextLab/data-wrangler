Overview
================

Datasets come in all shapes and sizes, and are often *messy*:

  - Observations come in different formats
  - There are missing values
  - Labels are missing and/or aren't consistent
  - Datasets need to be wrangled 🐄 🐑 🚜

The main goal of ``data-wrangler`` is to turn messy data into clean(er) data, defined as one of the following:

  - A single two-dimensional ``numpy`` ``array`` whose rows are observations and whose columns are features
  - A single ``pandas`` ``DataFrame`` whose indices are observations and whose columns are features
  - A list of ``array`` objects (each formatted as described above)
  - A list of ``DataFrame`` objects (each formatted as described above)


Usage
------

To use datawrangler in a project::

    import datawrangler as dw


Supported data formats
----------------------

One package can't accommodate every foreseeable format or input source, but ``data-wrangler`` provides a framework for adding support for new datatypes in a straightforward way.  Essentially, adding support for a new data type entails writing two functions:

  - An ``is_<datatype>`` function, which should return ``True`` if an object is compatible with the given datatype (or format), and ``False`` otherwise
  - A ``wrangle_<datatype>`` function, which should take in an object of the given type or format and return a ``pandas`` ``DataFrame`` with numerical entries

Currently supported datatypes are limited to:

  - ``array``-like objects
  - ``DataFrame``-like or ``Series``-like objects
  - text data (text is embedded using cutting-edge natural language processing
or lists of mixtures of the above.

Missing observations (e.g., ``nan``s, empty strings, etc.) may be filled in using:

  - Probabilistic principle component analysis
  - Interpolation to fill in observations with no features (e.g., when nearby observations are available)
