#!/usr/bin/env python

"""Tests for `datawrangler` package (core module)."""

import datawrangler as dw
import os

import pytest
from sklearn.feature_extraction.text import CountVectorizer


def test_get_default_options():
    defaults = dw.core.get_default_options()
    assert type(defaults) is dict

    keys = list(defaults.keys())

    assert 'CountVectorizer' in keys

    assert 'text' in keys
    assert eval(defaults['text']['model']) == ['CountVectorizer', 'LatentDirichletAllocation']

    assert 'data' in keys
    assert os.path.exists(eval(defaults['data']['homedir']))
    assert os.path.exists(eval(defaults['data']['datadir']))


def test_apply_defaults():
    defaults = dw.core.get_default_options()['CountVectorizer']

    cv1 = CountVectorizer().get_params()
    cv2 = dw.core.apply_defaults(CountVectorizer)().get_params()

    for k in defaults.keys():
        assert cv2[k] == eval(defaults[k])

    for k in cv1.keys():
        if k not in defaults.keys():
            assert cv1[k] == cv2[k]


def test_update_dict():
    d1 = {'a': 1, 'b': 2}
    d2 = {'a': 3, 'c': 4}

    d3 = dw.core.update_dict(d1, d2)

    assert d1['a'] == 1
    assert d1['b'] == 2

    assert d2['a'] == 3
    assert d2['c'] == 4

    assert d3['a'] == 3
    assert d3['b'] == 2
    assert d3['c'] == 4


def test_apply_defaults_unknown_name():
    """Regression: apply_defaults must not crash for names absent from config.ini.

    Previously raised KeyError, which broke every sklearn model without a config.ini
    section (e.g. TSNE, MDS, Isomap, SpectralEmbedding).
    """
    def not_in_config(a, b=2):
        return a, b

    wrapped = dw.core.apply_defaults(not_in_config)
    # no config section -> nothing injected, call passes through unchanged
    assert wrapped(1) == (1, 2)
    assert wrapped(1, b=5) == (1, 5)


def test_dataframe_backend_config():
    """set/get/reset of the global DataFrame backend, including validation."""
    from datawrangler.core.configurator import (
        set_dataframe_backend, get_dataframe_backend, reset_dataframe_backend)

    try:
        assert get_dataframe_backend() == 'pandas'  # default
        set_dataframe_backend('polars')
        assert get_dataframe_backend() == 'polars'
        with pytest.raises(ValueError):
            set_dataframe_backend('bogus')
        # a rejected value must not have mutated the current backend
        assert get_dataframe_backend() == 'polars'
    finally:
        reset_dataframe_backend()
    assert get_dataframe_backend() == 'pandas'


def test_version_is_single_sourced():
    """Regression for issue #29: __version__ derives from the package metadata (single source of truth),
    so it can never drift from setup.py's version."""
    from importlib.metadata import version, PackageNotFoundError
    try:
        assert dw.__version__ == version('pydata-wrangler')
    except PackageNotFoundError:  # uninstalled source checkout -> falls back to a literal
        assert isinstance(dw.__version__, str) and dw.__version__
