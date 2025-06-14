#!/usr/bin/env python

"""Tests for `datawrangler` package (decorate module)."""

import os
import datawrangler as dw
import pandas as pd
import polars as pl
import numpy as np
import pytest
from .conftest import assert_backend_type, assert_dataframes_equivalent


# noinspection PyTypeChecker
def test_list_generalizer():
    @dw.decorate.list_generalizer
    def f(x):
        return x ** 2

    assert f(3) == 9
    assert f([2, 3, 4]) == [4, 9, 16]


# noinspection PyTypeChecker
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_funnel(data_file, data, img_file, text_file, backend):
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
    text_kwargs = {'model': 'all-MiniLM-L6-v2'}
    wrangle_kwargs = {'return_dtype': True, 'backend': backend}
    wrangled, inferred_dtypes = f([data_file, data, data.values, img_file, text_file],
                                  dataframe_kwargs=dataframe_kwargs,
                                  text_kwargs=text_kwargs,
                                  wrangle_kwargs=wrangle_kwargs)

    correct_dtypes = ['dataframe', 'dataframe', 'array', 'array', 'text']
    assert all([i == c for i, c in zip(inferred_dtypes, correct_dtypes)])

    # Convert to numpy arrays for comparison (works with both backends)
    wrangled_0_values = wrangled[0].to_numpy() if hasattr(wrangled[0], 'to_numpy') else wrangled[0].values
    wrangled_1_values = wrangled[1].to_numpy() if hasattr(wrangled[1], 'to_numpy') else wrangled[1].values
    wrangled_2_values = wrangled[2].to_numpy() if hasattr(wrangled[2], 'to_numpy') else wrangled[2].values
    
    assert np.allclose(wrangled_0_values, wrangled_1_values)
    assert np.allclose(wrangled_0_values, wrangled_2_values)

    assert wrangled[3].shape == (1400, 5760)
    wrangled_3_values = wrangled[3].to_numpy() if hasattr(wrangled[3], 'to_numpy') else wrangled[3].values
    assert np.isclose(wrangled_3_values.mean(), 152.19, atol=0.1)
    assert dw.util.btwn(wrangled[3], 12, 248)

    assert wrangled[4].shape == (1, 384)  # all-MiniLM-L6-v2 produces 384-dim embeddings
    assert dw.util.btwn(wrangled[4], -1, 1)
    wrangled_4_values = wrangled[4].to_numpy() if hasattr(wrangled[4], 'to_numpy') else wrangled[4].values
    assert np.isclose(wrangled_4_values.mean(), -0.0007741971, atol=1e-5)
    
    # Verify backend types
    for w in wrangled:
        if dw.zoo.is_dataframe(w):
            assert_backend_type(w, backend)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_interpolate(data, backend):
    # Convert data to appropriate backend for testing
    if backend == 'polars':
        import polars as pl
        data = pl.from_pandas(data)
        
    # test imputing
    if backend == 'pandas':
        impute_test = data.copy()
        impute_test.loc[4, 'SecondDim'] = np.nan
        impute_test.loc[6, 'FourthDim'] = np.nan  # Fixed index
    else:  # polars
        # Polars uses .clone() instead of .copy()
        impute_test = data.clone()
        # Polars doesn't have .loc, use different approach
        impute_test = impute_test.with_columns([
            pl.when(pl.int_range(pl.len()) == 4).then(None).otherwise(pl.col('SecondDim')).alias('SecondDim'),
            pl.when(pl.int_range(pl.len()) == 6).then(None).otherwise(pl.col('FourthDim')).alias('FourthDim')  # Note: polars is 0-indexed
        ])

    @dw.decorate.interpolate
    def f(x):
        return x

    # noinspection PyCallingNonCallable
    recovered_data1 = f(impute_test, interp_kwargs={'impute_kwargs': {'model': 'IterativeImputer'}})
    
    # Convert to numpy for comparison
    data_values = data.to_numpy() if hasattr(data, 'to_numpy') else data.values
    recovered_1_values = recovered_data1.to_numpy() if hasattr(recovered_data1, 'to_numpy') else recovered_data1.values
    
    assert np.allclose(data_values, recovered_1_values)
    assert dw.zoo.is_dataframe(data)
    assert dw.zoo.is_dataframe(recovered_data1)
    assert_backend_type(recovered_data1, backend)

    # test interpolation
    if backend == 'pandas':
        interp_test = data.copy()
        interp_test.loc[5] = np.nan
    else:  # polars
        interp_test = data.clone()
        # For Polars, set row 5 to null
        interp_test = interp_test.with_columns([
            pl.when(pl.int_range(pl.len()) == 5).then(None).otherwise(pl.col(col)).alias(col)
            for col in interp_test.columns
        ])
    
    # noinspection PyCallingNonCallable
    recovered_data2 = f(interp_test, interp_kwargs={'method': 'linear'}, backend=backend)
    
    # Convert to numpy for comparison
    recovered_2_values = recovered_data2.to_numpy() if hasattr(recovered_data2, 'to_numpy') else recovered_data2.values
    
    assert np.allclose(data_values, recovered_2_values)
    assert dw.zoo.is_dataframe(data)
    assert dw.zoo.is_dataframe(recovered_data2)
    assert_backend_type(recovered_data2, backend)

    # impute + interpolate
    if backend == 'pandas':
        impute_interp_test = data.copy()
        impute_interp_test.loc[2, 'ThirdDim'] = np.nan
        impute_interp_test.loc[0, 'FourthDim'] = np.nan
        impute_interp_test.loc[6, 'FifthDim'] = np.nan  # Fixed index
        impute_interp_test.loc[4] = np.nan
    else:  # polars
        impute_interp_test = data.clone()
        # For Polars, set specific cells and entire row to null
        impute_interp_test = impute_interp_test.with_columns([
            pl.when(pl.int_range(pl.len()) == 2).then(None).otherwise(pl.col('ThirdDim')).alias('ThirdDim'),
            pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col('FourthDim')).alias('FourthDim'),
            pl.when(pl.int_range(pl.len()) == 6).then(None).otherwise(pl.col('FifthDim')).alias('FifthDim')
        ])
        # Set entire row 4 to null
        impute_interp_test = impute_interp_test.with_columns([
            pl.when(pl.int_range(pl.len()) == 4).then(None).otherwise(pl.col(col)).alias(col)
            for col in impute_interp_test.columns
        ])

    # noinspection PyCallingNonCallable
    recovered_data3 = f(impute_interp_test, interp_kwargs={'impute_kwargs': {'model': 'IterativeImputer'},
                                                           'method': 'pchip'}, backend=backend)
    
    # Convert to numpy for comparison
    recovered_3_values = recovered_data3.to_numpy() if hasattr(recovered_data3, 'to_numpy') else recovered_data3.values
    impute_interp_values = impute_interp_test.to_numpy() if hasattr(impute_interp_test, 'to_numpy') else impute_interp_test.values
    
    assert np.allclose(recovered_3_values[~np.isnan(impute_interp_values)],
                       data_values[~np.isnan(impute_interp_values)])
    assert dw.zoo.is_dataframe(data)
    assert dw.zoo.is_dataframe(recovered_data3)
    assert_backend_type(recovered_data3, backend)


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
