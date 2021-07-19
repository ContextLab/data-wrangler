import os
import pytest
import pandas as pd


@pytest.fixture
def resources():
    return os.path.join(os.path.split(os.path.dirname(__file__))[0], 'resources')


@pytest.fixture
def data_file(resources):
    return os.path.join(resources, 'testdata.csv')


@pytest.fixture
def data_url():
    return 'https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/testdata.csv'


@pytest.fixture
def img_file(resources):
    return os.path.join(resources, 'wrangler.jpg')


@pytest.fixture
def img_url():
    return 'https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/wrangler.jpg'


@pytest.fixture
def text_file(resources):
    return os.path.join(resources, 'home_on_the_range.txt')


@pytest.fixture
def text_url():
    return 'https://raw.githubusercontent.com/ContextLab/data-wrangler/main/tests/resources/home_on_the_range.txt'


@pytest.fixture
def data(data_file):
    return pd.read_csv(data_file, index_col=0)
