=======
History
=======

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
