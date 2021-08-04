#!/usr/bin/env python

"""Tests for `datawrangler` package (decorate module)."""

import os
import datawrangler as dw
import pandas as pd
import numpy as np


# noinspection PyTypeChecker
def test_list_generalizer():
    @dw.decorate.list_generalizer
    def f(x):
        return x ** 2

    assert f(3) == 9
    assert f([2, 3, 4]) == [4, 9, 16]


# noinspection PyTypeChecker
def test_funnel(data_file, data, img_file, text_file):
    @dw.decorate.list_generalizer
    @dw.decorate.funnel
    def g(x):
        return x.pow(2)

    assert int(g(3).values) == 9
    assert list([int(i.values) for i in g([3, 4, 5])]) == [9, 16, 25]
    assert g(np.array([1, 2, 3])).values.tolist() == [[1, 4, 9]]

    # noinspection PyShadowingNames
    @dw.decorate.funnel
    def f(x):
        assert dw.zoo.is_dataframe(x[0][0])
        assert dw.zoo.is_text(x[1][0])
        return x

    dataframe_kwargs = {'load_kwargs': {'index_col': 0}}
    text_kwargs = {'model': 'StackedEmbeddings'}
    wrangle_kwargs = {'return_dtype': True}
    wrangled, inferred_dtypes = f([data_file, data, data.values, img_file, text_file],
                                  dataframe_kwargs=dataframe_kwargs,
                                  text_kwargs=text_kwargs,
                                  wrangle_kwargs=wrangle_kwargs)

    correct_dtypes = ['dataframe', 'dataframe', 'array', 'array', 'text']
    assert all([i == c for i, c in zip(inferred_dtypes, correct_dtypes)])

    assert np.allclose(wrangled[0].values, wrangled[1].values)

    assert np.allclose(wrangled[0].values, wrangled[2].values)

    assert wrangled[3].shape == (1400, 5760)
    assert np.isclose(wrangled[3].values.mean(), 152.193)
    assert dw.util.btwn(wrangled[3], 12, 248)

    assert wrangled[4].shape == (1, 4196)
    assert dw.util.btwn(wrangled[4], -1, 1)
    assert np.isclose(wrangled[4].values.mean(), 0.00449942)


def test_interpolate(data):
    # test imputing
    impute_test = data.copy()
    impute_test.loc[4, 'SecondDim'] = np.nan
    impute_test.loc[8, 'FourthDim'] = np.nan

    @dw.decorate.interpolate
    def f(x):
        return x

    # noinspection PyCallingNonCallable
    recovered_data1 = f(impute_test, interp_kwargs={'impute_kwargs': {'model': 'IterativeImputer'}})
    assert np.allclose(data, recovered_data1)
    assert dw.zoo.is_dataframe(data)
    assert dw.zoo.is_dataframe(recovered_data1)

    # test interpolation
    interp_test = data.copy()
    interp_test.loc[5] = np.nan
    # noinspection PyCallingNonCallable
    recovered_data2 = f(interp_test, interp_kwargs={'method': 'linear'})
    assert np.allclose(data, recovered_data2)
    assert dw.zoo.is_dataframe(data)
    assert dw.zoo.is_dataframe(recovered_data2)

    # impute + interpolate
    impute_interp_test = data.copy()
    impute_interp_test.loc[2, 'ThirdDim'] = np.nan
    impute_interp_test.loc[0, 'FourthDim'] = np.nan
    impute_interp_test.loc[8, 'FifthDim'] = np.nan
    impute_interp_test.loc[4] = np.nan

    # noinspection PyCallingNonCallable
    recovered_data3 = f(impute_interp_test, interp_kwargs={'impute_kwargs': {'model': 'IterativeImputer'},
                                                           'method': 'pchip'})
    assert np.allclose(recovered_data3.values[~np.isnan(impute_interp_test)],
                       data.values[~np.isnan(impute_interp_test)])
    assert dw.zoo.is_dataframe(data)
    assert dw.zoo.is_dataframe(recovered_data3)


def test_apply_unstacked(data):
    i = 3
    data1 = data.iloc[:i]
    data2 = data.iloc[i:]
    stacked_data = dw.stack([data1, data2])

    assert np.allclose(stacked_data, data)

    @dw.decorate.apply_unstacked
    def f(x):
        return pd.DataFrame(x.mean(axis=0)).T

    means = f(stacked_data)
    assert dw.zoo.is_multiindex_dataframe(means)
    assert np.allclose(means.iloc[0], data1.mean(axis=0))
    assert np.allclose(means.iloc[1], data2.mean(axis=0))

    xs = [np.cumsum(np.random.randn(100, 5), axis=0) for _ in range(10)]

    @dw.decorate.apply_unstacked
    def g(x):
        return x

    assert all(np.allclose(x, y) for x, y in zip(xs, g(xs)))
    assert all(np.allclose(x, y) for x, y in zip(xs, dw.unstack(g(dw.stack(xs)))))


def test_unstack(data):
    xs = dw.wrangle([np.cumsum(np.random.randn(100, 5), axis=0) for _ in range(10)])
    ys = dw.wrangle([np.cumsum(np.random.randn(100, 5), axis=0) for _ in range(10)])

    stacked_xs = dw.stack(xs)
    stacked_ys = dw.stack(ys)
    stacked_xy = dw.stack([stacked_xs, stacked_ys])

    assert np.allclose(dw.unstack(stacked_xs)[0], xs[0])
    assert np.allclose(dw.unstack(stacked_xs)[0].index.values, xs[0].index.values)

    assert np.allclose(dw.unstack(stacked_xy)[0], stacked_xs)
    assert dw.zoo.is_multiindex_dataframe(dw.unstack(stacked_xy)[0])
    assert np.allclose(dw.unstack(stacked_xy)[0].index.to_frame(), stacked_xs.index.to_frame())


def test_apply_stacked(data):
    i = 4
    data1 = data.iloc[:i]
    data2 = data.iloc[i:]

    @dw.decorate.apply_stacked
    def f(x):
        return x.mean(axis=0)

    # noinspection PyTypeChecker
    means = f([data1, data2])
    assert np.allclose(means, data.mean(axis=0))
