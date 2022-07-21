#!/usr/bin/env python

"""Tests for `datawrangler` package (core module)."""

import datawrangler as dw
import configparser
import os

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
