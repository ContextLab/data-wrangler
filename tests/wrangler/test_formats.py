#!/usr/bin/env python

"""Tests for `datawrangler` package."""

import datawrangler as dw
import numpy as np
import pandas as pd
import os

resources = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'resources')
data_file = os.path.join(resources, 'testdata.csv')
img_file = os.path.join(resources, 'wrangler.jpg')
text_file = os.path.join(resources, 'home_on_the_range.txt')

data = pd.read_csv(data_file, index_col=0)


def test_is_dataframe():
    assert dw.zoo.is_dataframe(data)
    assert dw.zoo.is_dataframe(pd.DataFrame(np.zeros([10, 3])))
    assert not dw.zoo.is_dataframe(img_file)
    assert not dw.zoo.is_dataframe(text_file)


def test_dataframe_like():
    assert dw.zoo.dataframe_like(data)
    assert not dw.zoo.dataframe_like(img_file)


def test_wrangle_dataframe():
    assert dw.zoo.is_dataframe(dw.zoo.wrangle_dataframe(data))

    df = dw.zoo.wrangle_dataframe(data)
    assert df.index.name == 'ByTwos'
    assert np.all(df['FirstDim'] == np.arange(1, 8))
    assert np.all(df['SecondDim'] == np.arange(2, 16, 2))
    assert np.all(df['ThirdDim'] == np.arange(3, 24, 3))
    assert np.all(df['FourthDim'] == np.arange(4, 32, 4))
    assert np.all(df['FifthDim'] == np.arange(5, 40, 5))


def test_is_array():
    assert dw.zoo.is_array(data.values)
    assert not dw.zoo.is_array(img_file)
    assert not dw.zoo.is_array(text_file)


def test_wrangle_array():
    df = dw.zoo.wrangle_array(data.values)
    assert dw.zoo.is_dataframe(df)
    assert df.shape == (7, 5)


def test_get_image():
    img = dw.zoo.image.get_image(img_file)
    assert img is not None
    assert img.shape == (1400, 1920, 3)
    assert np.max(img) == 248
    assert np.min(img) == 12
    assert np.isclose(np.mean(img), 152.193)


def test_is_image():
    assert dw.zoo.is_image(img_file)


def test_wrangle_image():
    df = dw.zoo.wrangle_image(img_file)
    assert df.shape == (1400, 5760)
    assert dw.zoo.is_dataframe(df)
    assert np.max(df.values) == 248
    assert np.min(df.values) == 12
    assert np.isclose(np.mean(df.values), 152.193)


def test_load_text():
    text = dw.zoo.text.load_text(text_file).split('\n')
    assert text[0] == 'O give me a home where the buffaloes roam'
    assert text[-1] == 'And the skies are not cloudy all day'


def test_is_text():
    assert dw.zoo.is_text(text_file)
    assert not dw.zoo.is_text(img_file)
    assert not dw.zoo.is_text(data_file)


def test_get_corpus():
    # test sotus corpus (small)

    # test small hugging face corpus: cbt/raw
    pass


def test_wrangle_text():
    # scikit-learn CountVectorizer

    # scikit-learn CountVectorizer + LatentDirichletAllocation

    # Hugging Face
    pass


# TODO:
#   - wrangle text with various models and corpora
#   - other text functions
#   - is_null
#   - wrangle_null
#   - decorators
#   - io
#   - ppca
#   - interpolation
#   - helper functions
#   - divide tests into separate files:
#      - one per datatype
#      - one per additional function beyond datatype-specific formatting
