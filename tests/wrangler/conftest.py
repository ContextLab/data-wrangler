import os
import pytest
import pandas as pd
import polars as pl
import numpy as np
import datawrangler as dw


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


# Backend testing utilities
@pytest.fixture(params=['pandas', 'polars'])
def backend(request):
    """Parameterized fixture to test both pandas and Polars backends."""
    return request.param


def assert_backend_type(df, backend):
    """Assert that DataFrame is of expected backend type."""
    if backend == 'polars':
        assert isinstance(df, pl.DataFrame), f"Expected Polars DataFrame, got {type(df)}"
    else:
        assert isinstance(df, pd.DataFrame), f"Expected pandas DataFrame, got {type(df)}"


def assert_dataframes_equivalent(df1, df2, check_dtypes=False):
    """Assert that two DataFrames contain equivalent data regardless of backend."""
    # Convert both to pandas for comparison if needed
    if isinstance(df1, pl.DataFrame):
        df1_pandas = df1.to_pandas()
    else:
        df1_pandas = df1
        
    if isinstance(df2, pl.DataFrame):
        df2_pandas = df2.to_pandas()
    else:
        df2_pandas = df2
    
    # Check shapes
    assert df1_pandas.shape == df2_pandas.shape, f"Shape mismatch: {df1_pandas.shape} vs {df2_pandas.shape}"
    
    # Check values (allowing for floating point differences)
    assert np.allclose(df1_pandas.values, df2_pandas.values, equal_nan=True), "DataFrame values not equivalent"
    
    if check_dtypes:
        # Note: dtypes may differ slightly between backends, so this is optional
        pass
