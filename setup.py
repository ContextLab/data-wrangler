# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install


NAME = 'datawrangler'
VERSION = '0.1.0'
AUTHOR = 'Contextual Dynamics Lab'
AUTHOR_EMAIL = 'contextualdynamics@gmail.com'
URL = 'https://github.com/ContextLab/data-wrangler'
DOWNLOAD_URL = URL
LICENSE = 'MIT'
REQUIRES_PYTHON = '>=3.6'
PACKAGES = find_packages(exclude=('images', 'examples', 'tests'))
with open('requirements.txt', 'r') as f:
    REQUIREMENTS = f.read().splitlines()

DESCRIPTION = 'Wrangle your messy data into consistent well-organized formats!'
LONG_DESCRIPTION = """\
The main goal of data-wrangler is to turn messy data into clean(er) data, defined as one of the following:

  - A single two-dimensional numpy array whose rows are observations and whose columns are features
  - A single pandas DataFrame whose indices are observations and whose columns are features
  - A list of arrays (each formatted as described above)
  - A list of DataFrames (each formatted as described above)
"""
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS'
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    python_requires=REQUIRES_PYTHON,
    packages=PACKAGES,
    install_requires=REQUIREMENTS,
    classifiers=CLASSIFIERS
)
