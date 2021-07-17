#!/usr/bin/env python

"""Tests for `datawrangler` package (util module)."""

import datawrangler as dw
import numpy as np


def test_btwn():
    assert dw.util.btwn(np.array([1, 2, 3, 4, 5, -10]), -10, 5)
    assert not dw.util.btwn(np.array([2, 3, 4]), 3.2, 3.3)
    assert dw.util.btwn(np.array([10, -100, 1000, -10000]), -10000, 1000)


def test_dataframe_like(data, data_file, img_file, text_file):
    assert dw.util.dataframe_like(data)
    assert not dw.util.dataframe_like(data_file)
    assert not dw.util.dataframe_like(img_file)
    assert not dw.util.dataframe_like(text_file)
    assert not dw.util.dataframe_like(data.values)


def test_array_like(data, data_file, img_file, text_file):
    assert dw.util.array_like(data)
    assert dw.util.array_like(data_file)

    assert dw.util.array_like(img_file)

    assert not dw.util.array_like(text_file)
    assert not dw.util.array_like('test')

    assert dw.util.array_like(data.values)
    assert dw.util.array_like([])
    assert dw.util.array_like([1, 2, 3])
    assert dw.util.array_like(['one', 'two', 'three'])
    assert dw.util.array_like(np.arange(10))


def test_depth():
    assert dw.util.depth([]) == 0
    assert dw.util.depth([1]) == 1
    assert dw.util.depth([1, 2, 3]) == 1
    assert dw.util.depth([1, [2, 3]]) == 2
    assert dw.util.depth([[1], [2, 3], [4, [5, 6]]]) == 3
