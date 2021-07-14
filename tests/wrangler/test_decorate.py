#!/usr/bin/env python

"""Tests for `datawrangler` package (decorate module)."""

import os
import datawrangler as dw
import pandas as pd

from dataloader import resources, data_file, img_file, text_file, data


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
        return [x.index, x.columns]

    x = f([data_file, data, img_file, text_file])
    pass


test_funnel()


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
