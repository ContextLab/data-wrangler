#!/usr/bin/env python

"""Tests for `datawrangler` package (io module)."""

import datawrangler as dw
import numpy as np


def test_load(data_file, img_file, text_file):
    data = dw.io.load(data_file)
    image = dw.io.load(img_file)
    text = dw.io.load(text_file)

    assert dw.zoo.is_dataframe(data)
    assert not dw.zoo.is_dataframe(image)
    assert not dw.zoo.is_dataframe(text)

    assert not dw.zoo.is_array(data)
    assert dw.zoo.is_array(image)
    assert not dw.zoo.is_array(text)

    assert not dw.zoo.is_text(data)
    assert not dw.zoo.is_text(image)
    assert dw.zoo.is_text(text)


# noinspection PyUnusedLocal
def test_save(data_file, data_url, img_file, img_url, text_file, text_url):
    for dtype in ['data', 'img', 'text']:
        local = dw.io.load(eval(f'{dtype}_file'))
        remote = dw.io.load(eval(f'{dtype}_url'))  # requires downloading and saving the remote file

        assert np.all(local == remote)
