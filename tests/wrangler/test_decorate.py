#!/usr/bin/env python

"""Tests for `datawrangler` package (decorate module)."""

import os
import datawrangler as dw
import pandas as pd
import numpy as np

from .dataloader import resources, data_file, img_file, text_file, data


# noinspection PyTypeChecker
def test_list_generalizer():
    @dw.decorate.list_generalizer
    def f(x):
        return x ** 2

    assert f(3) == 9
    assert f([2, 3, 4]) == [4, 9, 16]


# noinspection PyTypeChecker
def test_funnel():
    # noinspection PyShadowingNames
    @dw.decorate.funnel
    def f(x):
        return x

    dataframe_kwargs = {'load_kwargs': {'index_col': 0}}
    text_kwargs = {'model': 'StackedEmbeddings'}
    wrangle_kwargs = {'return_dtype': True}
    wrangled, inferred_dtypes = f([data_file, data, img_file, text_file],
                                  dataframe_kwargs=dataframe_kwargs,
                                  text_kwargs=text_kwargs,
                                  wrangle_kwargs=wrangle_kwargs)

    correct_dtypes = ['dataframe', 'dataframe', 'image', 'text']
    assert all([i == c for i, c in zip(inferred_dtypes, correct_dtypes)])

    assert np.allclose(wrangled[0].values, wrangled[1].values)

    assert wrangled[2].shape == (1400, 5760)
    assert np.isclose(wrangled[2].values.mean(), 152.193)
    assert dw.util.btwn(wrangled[2], 12, 248)

    assert wrangled[3].shape == (1, 4196)
    assert dw.util.btwn(wrangled[3], -1, 1)
    assert np.isclose(wrangled[3].values.mean(), 0.00449942)

    # TODO: also double check that funnel's decoration with list_generalizer is working as expected


def test_fill_missing():
    pass


def test_interpolate():
    pass


def test_stack_handler():
    pass


def test_module_checker():
    pass


def test_unstack_apply():
    pass


def test_stack_apply():
    pass
