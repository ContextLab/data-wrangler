import os
import requests
import dill
import numpy as np
from hashlib import blake2b as hasher
from matplotlib import pyplot as plt

from ..core.configurator import get_default_options
from .panda_handler import load_dataframe
from .extension_handler import get_extension

defaults = get_default_options()
img_types = plt.gcf().canvas.get_supported_filetypes().keys();


def get_local_fname(x, digest_size=10):
    """
    Internal data-wrangler function for generating filenames for saved datasets

    Parameters
    ----------
    :param x: a string containing some data
    :param digest_size: length of the hash to compute (default: 10)

    Returns
    -------
    :return: The absolute path of the location where the given information should be stored.
    """
    if os.path.exists(x):
        return x

    h = hasher(digest_size=digest_size)
    h.update(x.encode('ascii'))
    return os.path.join(defaults['data']['datadir'], h.hexdigest() + '.' + get_extension(x))


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def load_remote(url, params=None):
    if params is None:
        params = {}
    session = requests.Session()
    response = session.get(url, params=params, stream=True)
    token = get_confirm_token(response)
    if token:
        params['confirm'] = token
        response = session.get(url, params=params, stream=True)

    if get_extension(url) in ['txt']:
        return response.text
    else:
        return response.content


def load(x, base_url='https://docs.google.com/uc?export=download', dtype=None, **kwargs):
    """
    Load local or remote files in a wide range of formats

    Parameters
    ----------
    :param x: a string containing a URL, document ID (e.g., a Google object's ID), or file path
    :param base_url: a template URL used to download objects specified using only their ID.
      Default: 'https://docs.google.com/uc?export=download' (supports Google IDs)
    :param dtype: Optional argument for specifying how the data should be loaded; can be one of:
      - 'pickle': use the dill library to load in pickled objects and functions
      - 'numpy': treat the dataset as a .npy or .npz file
      - None (default): attempt to determine the filetype automatically based on the URL or file extension.  The
        following filetypes are supported:
          - txt files: treated as plain text
          - any filetype supported by the Pandas library:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
          - any image filetype supported by the Matplotlib library; for a full list see:
            matplotlib.pyplot.gcf().canvas.get_supported_filetypes()
    :param kwargs: any additional keyword arguments are passed to whatever function is selected to load in the dataset.  For
      example, when loading in a csv file (a Pandas-compatible format), passing the keyword argument index_col=0 will
      tell Pandas to interpret the first (0) column as the resulting DataFrame's index when loading the file's contents
      into a DataFrame.

    Returns
    -------
    :return: the retrieved data.  Remote files will be cached (saved) locally to disk for faster loading if/when the same
    address or ID is used to load the file again at a later time.
    """
    # noinspection PyShadowingNames
    def helper(fname, **helper_kwargs):
        if dtype == 'pickle':
            with open(fname, 'rb') as f:
                return dill.load(f, **helper_kwargs)
        elif dtype == 'numpy':
            if 'allow_pickle' not in helper_kwargs.keys():
                helper_kwargs['allow_pickle'] = True
            data = np.load(fname, **helper_kwargs)
            if type(data) is dict:
                if len(data.keys()) == 1:
                    return data[list(data.keys())[0]]
            return data
        else:
            ext = get_extension(fname)
            if ext == 'txt':
                with open(fname, 'r') as f:
                    return ''.join(f.readlines())
            elif ext in ['csv', 'xls', 'xlsx', 'json', 'html', 'xml', 'hdf', 'feather', 'parquet', 'orc', 'sas',
                         'spss', 'sql', 'gbq', 'stata', 'pkl']:
                return load_dataframe(fname, **kwargs)
            elif ext in ['npy', 'npz']:
                return np.load(fname)
            elif ext in img_types:
                return plt.imread(fname)
            else:
                raise ValueError(f'Unknown datatype: {dtype}')

    assert type(x) is str, IOError('cannot interpret non-string filename')
    # FIXME: remote images aren't loaded correctly...
    if os.path.exists(x):
        return helper(x, **kwargs)

    fname = get_local_fname(x)
    if os.path.exists(fname):
        return helper(fname, **kwargs)
    else:
        if x.startswith('http'):
            data = load_remote(x)
        else:
            # noinspection PyBroadException
            try:     # is x a Google ID?
                data = load_remote(base_url, params={'id': x})
            except:
                raise IOError('cannot find data at source: {x}')
        save(x, data, dtype=dtype)
        return load(x, dtype=dtype, **kwargs)


def save(x, obj, dtype=None, **kwargs):
    """
    Save data to disk.

    Parameters
    ----------
    :param x: the file's original path, URL, or ID (used to create a hash to define a new filename)
    :param obj: the data to store to disk
    :param dtype: optional argument specifying how to store the data; can be one of:
      - 'pickle': use the dill library to pickle the object
      - 'numpy': save the objects as a compressed (.npz-formatted) numpy file
      - None (default): determine the filetype automatically; if x is passed in as bytes, write x directly to disk. If
        x is a string, treat x as text.
    :param kwargs: any additional keyword arguments are passed to dill.dump (if dtype == 'pickle') or numpy.savez (if
        dtype == 'numpy').  For any other datatype, additional keyword arguments are ignored.

    Returns
    -------
    :return: None
    """
    assert type(x) is str, IOError('cannot interpret non-string filename')
    fname = get_local_fname(x)

    if type(obj) is bytes:
        with open(fname, 'wb') as f:
            f.write(obj)
    elif type(obj) is str:
        with open(fname, 'w') as f:
            f.write(obj)
    elif dtype == 'pickle':
        with open(fname, 'wb') as f:
            dill.dump(obj, f, **kwargs)
    elif dtype == 'numpy':
        np.savez(fname, obj, **kwargs)
    else:
        raise ValueError(f'cannot save object (specified dtype: {dtype}; observed type: {type(obj)})')
