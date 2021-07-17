#!/usr/bin/env python

"""Tests for `datawrangler` package (io module)."""

import datawrangler as dw


def test_load(data_file, img_file, text_file):
    data = dw.io.load(data_file)
    image = dw.io.load(img_file)
    text = dw.io.load(text_file)

    assert dw.zoo.is_dataframe(data)
    assert not dw.zoo.is_dataframe(image)
    assert not dw.zoo.is_dataframe(text)

    assert not dw.zoo.is_image(data)
    assert dw.zoo.is_image(image)
    assert not dw.zoo.is_image(text)

    assert not dw.zoo.is_text(data)
    assert not dw.zoo.is_text(image)
    assert dw.zoo.is_text(text)


def test_save(text_file):
    remote_file = 'https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/home_on_the_range' \
                  '.txt'

    local_text = dw.io.load(text_file)
    remote_text = dw.io.load(remote_file)  # calls save and then loads the local version

    assert local_text == remote_text
