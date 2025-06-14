.. highlight:: shell

============
Installation
============

Requirements
------------

- **Python 3.9+** (v0.3.0+ requires modern Python versions)
- NumPy 2.0+ and pandas 2.0+ compatible
- Optional: HuggingFace transformers for advanced text processing

Stable release
--------------

**Basic Installation**

To install datawrangler, run this command in your terminal:

.. code-block:: console

    $ pip install pydata-wrangler

This installs the core functionality including sklearn-based text processing.

**Full Installation with ML Libraries**

For advanced text processing with sentence-transformers models:

.. code-block:: console

    $ pip install "pydata-wrangler[hf]"

This includes sentence-transformers, transformers, and related HuggingFace libraries.

**Upgrade from Previous Versions**

If upgrading from v0.2.x, ensure you have Python 3.9+:

.. code-block:: console

    $ pip install --upgrade "pydata-wrangler[hf]"

This is the preferred method to install datawrangler, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for datawrangler can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/ContextLab/data-wrangler

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/ContextLab/data-wrangler/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/ContextLab/data-wrangler
.. _tarball: https://github.com/ContextLab/data-wrangler/tarball/master
