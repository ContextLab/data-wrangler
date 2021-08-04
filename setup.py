#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

test_requirements = ['pytest>=3', ]

setup(
    author="Contextual Dynamics Lab",
    author_email='contextualdynamics@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Wrangle messy data into pandas DataFrames, with a special focus on text data and natural language "
                "processing",
    install_requires=requirements,
    license="MIT license",
    long_description='For more information see https://data-wrangler.readthedocs.io/en/latest/',
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords=['datawrangler', 'pydata-wrangler', 'python', 'pandas', 'natural language processing', 'image',
              'data science', 'data manipulation'],
    name='pydata-wrangler',
    packages=find_packages(include=['datawrangler', 'datawrangler.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ContextLab/data-wrangler',
    version='0.1.3',
    zip_safe=False,
)
