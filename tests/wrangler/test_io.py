#!/usr/bin/env python

"""Tests for `datawrangler` package (io module)."""

import os
import glob

import pytest
import datawrangler as dw
import numpy as np

from datawrangler.io.extension_handler import get_extension
from datawrangler.io.io import get_local_fname


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


def test_get_extension_strips_url_query_string():
    """URL query strings/fragments must not leak into the detected extension.

    Regression test: Dropbox/Google-Drive style links (e.g. '...file.npz?dl=1')
    previously produced the extension 'npz?dl=1', which polluted cache filenames
    and made the cached copy unreadable ('Unknown datatype: npz?dl=1').
    """
    # local paths keep working exactly as before
    assert get_extension('/some/dir/testdata.csv') == 'csv'
    assert get_extension('archive.tar.gz') == 'gz'
    assert get_extension('/no/extension/here') == 'dw'

    # remote URLs with query strings / fragments resolve to the TRUE extension
    assert get_extension('https://www.dropbox.com/s/abc/minipedia.npz?dl=1') == 'npz'
    assert get_extension('https://example.com/data.csv?dl=0') == 'csv'
    assert get_extension('https://example.com/data.json?a=1&b=2') == 'json'
    assert get_extension('https://example.com/pic.png#section') == 'png'
    # a query string that itself contains a dotted path must not fool detection
    assert get_extension('https://example.com/data.csv?redirect=/x/y.zip') == 'csv'


def test_get_local_fname_query_string_cache_name():
    """Cache filenames derived from query-string URLs use a clean extension."""
    url = 'https://www.dropbox.com/s/abc/minipedia.npz?dl=1'
    fname = get_local_fname(url)
    assert fname.endswith('.npz'), f'cache filename should end in .npz, got {fname!r}'
    assert '?' not in fname and '#' not in fname


def _isolated_cache(monkeypatch, tmp_path):
    """Point the datawrangler cache at an isolated temp dir and return it."""
    monkeypatch.setenv('HOME', str(tmp_path))
    cache = tmp_path / '.datawrangler' / 'data'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def test_remote_load_is_cached_once_and_reused(monkeypatch, tmp_path, data_url):
    """A remote file is cached exactly once and re-reads hit that same file.

    Regression test for the caching logic: re-loading must NOT re-download or
    create additional cache entries under new hashes.
    """
    cache = _isolated_cache(monkeypatch, tmp_path)
    try:
        first = dw.io.load(data_url)
    except Exception as e:  # pragma: no cover - network unavailable
        pytest.skip(f'network unavailable: {e}')

    files_after_first = sorted(glob.glob(str(cache / '*')))
    assert len(files_after_first) == 1, files_after_first

    second = dw.io.load(data_url)
    files_after_second = sorted(glob.glob(str(cache / '*')))
    assert files_after_second == files_after_first, 'reload created a duplicate cache entry'
    assert np.all(first.values == second.values)


def test_query_string_url_round_trips(monkeypatch, tmp_path, data_url):
    """A URL with a '?dl=1'-style query string downloads, caches, and reloads.

    Previously this raised 'ValueError: Unknown datatype: csv?dl=1'.
    """
    cache = _isolated_cache(monkeypatch, tmp_path)
    qs_url = data_url + '?dl=1'
    try:
        df = dw.io.load(qs_url)
    except Exception as e:  # pragma: no cover - network unavailable
        pytest.skip(f'network unavailable: {e}')

    assert dw.zoo.is_dataframe(df)
    cached = sorted(glob.glob(str(cache / '*')))
    assert len(cached) == 1 and cached[0].endswith('.csv'), cached


def test_load_values(data_file):
    """Deeper than boolean type checks: verify the actual loaded CSV content."""
    data = dw.io.load(data_file, index_col=0)
    assert dw.zoo.is_dataframe(data)
    assert data.shape == (7, 5)
    assert list(data.columns) == ['FirstDim', 'SecondDim', 'ThirdDim', 'FourthDim', 'FifthDim']
    # concrete known values from tests/resources/testdata.csv
    assert data['FirstDim'].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert data['FifthDim'].tolist() == [5, 10, 15, 20, 25, 30, 35]
    assert data.iloc[0].tolist() == [1, 2, 3, 4, 5]
    assert list(data.index) == [0, 2, 4, 5, 6, 8, 10]


def test_save_load_roundtrip_pickle(monkeypatch, tmp_path):
    """dw.io.save/load round-trips an arbitrary object via the 'pickle' dtype (real files)."""
    _isolated_cache(monkeypatch, tmp_path)
    obj = {'nested': [1, 2, 3], 'label': 'wrangler', 'arr': np.arange(4).tolist()}
    key = 'testkey://obj.pkl'
    dw.io.save(key, obj, dtype='pickle')

    cached = get_local_fname(key)
    assert os.path.exists(cached), 'save() did not write to the expected cache path'
    assert dw.io.load(cached, dtype='pickle') == obj


def test_save_load_roundtrip_numpy(monkeypatch, tmp_path):
    """dw.io.save/load round-trips an array via the 'numpy' dtype (real files)."""
    _isolated_cache(monkeypatch, tmp_path)
    arr = np.arange(12).reshape(3, 4)
    key = 'testkey://arr.npz'
    dw.io.save(key, arr, dtype='numpy')

    cached = get_local_fname(key)
    assert os.path.exists(cached)
    loaded = dw.io.load(cached, dtype='numpy')
    try:
        assert np.allclose(loaded['arr_0'], arr)   # np.savez stores a positional array as 'arr_0'
    finally:
        loaded.close()
