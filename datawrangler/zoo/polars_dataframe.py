"""
Polars DataFrame support for data-wrangler.

This module provides functions for detecting and wrangling Polars DataFrames,
as well as conversion utilities between pandas and Polars.
"""

from ..util.lazy_imports import lazy_import

# Lazy import Polars
get_polars = lazy_import('polars')


def is_polars_dataframe(x):
    """
    Determine if an object is a Polars DataFrame.
    
    Parameters
    ----------
    x : object
        The object to check
        
    Returns
    -------
    bool
        True if the object is a Polars DataFrame, False otherwise
    """
    try:
        pl = get_polars()
        return isinstance(x, pl.DataFrame)
    except ImportError:
        return False


def is_polars_lazyframe(x):
    """
    Determine if an object is a Polars LazyFrame.
    
    Parameters
    ----------
    x : object
        The object to check
        
    Returns
    -------
    bool
        True if the object is a Polars LazyFrame, False otherwise
    """
    try:
        pl = get_polars()
        return isinstance(x, pl.LazyFrame)
    except ImportError:
        return False


def polars_to_pandas(df):
    """
    Convert a Polars DataFrame to a pandas DataFrame.
    
    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
        The Polars DataFrame to convert
        
    Returns
    -------
    pandas.DataFrame
        The converted pandas DataFrame
    """
    pl = get_polars()
    
    if is_polars_lazyframe(df):
        # Collect LazyFrame to DataFrame first
        df = df.collect()
    
    if not is_polars_dataframe(df):
        raise TypeError(f"Expected Polars DataFrame, got {type(df)}")
    
    return df.to_pandas()


def pandas_to_polars(df):
    """
    Convert a pandas DataFrame to a Polars DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The pandas DataFrame to convert
        
    Returns
    -------
    polars.DataFrame
        The converted Polars DataFrame
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    
    pl = get_polars()
    return pl.from_pandas(df)


def wrangle_polars_dataframe(data, return_model=False, **kwargs):
    """
    Wrangle a Polars DataFrame.
    
    This function accepts Polars DataFrames and LazyFrames and applies any
    specified transformations while preserving the Polars format.
    
    Parameters
    ----------
    data : polars.DataFrame or polars.LazyFrame
        The Polars DataFrame to wrangle
    return_model : bool, optional
        If True, return a function for transforming DataFrames along with
        the wrangled DataFrame. Default: False
    **kwargs : dict
        Additional keyword arguments passed to the wrangling model
        
    Returns
    -------
    polars.DataFrame or tuple
        The wrangled Polars DataFrame (if return_model is False), or a tuple
        of (DataFrame, model) if return_model is True
    """
    pl = get_polars()
    
    # Handle LazyFrames by collecting them
    if is_polars_lazyframe(data):
        data = data.collect()
    
    if not is_polars_dataframe(data):
        raise TypeError(f"Expected Polars DataFrame, got {type(data)}")
    
    # Extract model from kwargs or use default
    model = kwargs.pop('model', None)
    if model is None:
        model = {'model': pl.DataFrame, 'args': [], 'kwargs': kwargs}
    elif not isinstance(model, dict):
        model = {'model': model, 'args': [], 'kwargs': kwargs}
    
    # Apply the model (for now, just return the DataFrame)
    # In the future, this could apply Polars-specific transformations
    wrangled = data
    
    if return_model:
        return wrangled, model
    return wrangled


def create_polars_dataframe(data, columns=None):
    """
    Create a Polars DataFrame from various data types.
    
    Parameters
    ----------
    data : array-like, dict, or scalar
        The data to convert to a Polars DataFrame
    columns : list of str, optional
        Column names for the DataFrame
        
    Returns
    -------
    polars.DataFrame
        The created Polars DataFrame
    """
    pl = get_polars()
    
    # Handle different input types
    if isinstance(data, dict):
        return pl.DataFrame(data)
    elif hasattr(data, '__array__'):
        # NumPy array or similar
        import numpy as np
        arr = np.asarray(data)
        
        if arr.ndim == 1:
            # 1D array - create single column
            col_name = columns[0] if columns else "0"
            return pl.DataFrame({col_name: arr})
        elif arr.ndim == 2:
            # 2D array - create multiple columns
            if columns is None:
                columns = [str(i) for i in range(arr.shape[1])]
            return pl.DataFrame({col: arr[:, i] for i, col in enumerate(columns)})
        else:
            raise ValueError(f"Cannot create DataFrame from {arr.ndim}D array")
    else:
        # Try to create directly
        return pl.DataFrame(data)