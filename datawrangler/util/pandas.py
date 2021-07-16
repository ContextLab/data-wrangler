import numpy as np
import pandas as pd
from ..decorate import funnel
from ..zoo.dataframe import is_dataframe, is_multiindex_dataframe


@funnel
def pandas_stack(data, names=None, keys=None, verify_integrity=False, sort=False, copy=True, ignore_index=False,
                 levels=None):
    """
    Take a list of DataFrames with the same number of columns and (optionally)
    a list of names (of the same length as the original list; default:
    range(len(x))).  Return a single MultiIndex DataFrame where the original
    DataFrames are stacked vertically, with the data names as their level 1
    indices and their original indices as their level 2 indices.

    INPUTS
    data: data in any format supported by datawrangler

    Also takes all keyword arguments from pandas.concat except axis, join, join_axes

    All other keyword arguments (if any) are passed to funnel

    OUTPUTS
    a single MultiIndex DataFrame
    """

    if is_multiindex_dataframe(data):
        return data
    elif is_dataframe(data):
        data = [data]
    elif len(data) == 0:
        return None

    assert len(np.unique([d.shape[1] for d in data])) == 1, 'All DataFrames must have the same number of columns'
    for i, d1 in enumerate(data):
        template = d1.columns.values
        for d2 in data[(i + 1):]:
            assert np.all([(c in template) for c in d2.columns.values]), 'All DataFrames must have the same columns'

    if keys is None:
        keys = np.arange(len(data), dtype=int)

    assert is_array(keys) or (type(keys) == list), f'keys must be None or a list or array of length len(data)'
    assert len(keys) == len(data), f'keys must be None or a list or array of length len(data)'

    if names is None:
        names = ['ID', *[f'ID{i}' for i in range(1, len(data[0].index.names))], None]

    return pd.concat(data, axis=0, join='outer', names=names, keys=keys,
                     verify_integrity=verify_integrity, sort=sort, copy=copy,
                     ignore_index=ignore_index, levels=levels)


def pandas_unstack(x):
    if not is_multiindex_dataframe(x):
        if is_dataframe(x):
            return x
        else:
            raise Exception(f'Unsupported datatype: {type(x)}')

    names = list(x.index.names)
    grouper = 'ID'
    if not (grouper in names):
        names[0] = grouper
    elif not (names[0] == grouper):
        for i in np.arange(
                len(names)):    # trying n things other than 'ID'; one must be outside of the n-1 remaining names
            next_grouper = f'{grouper}{i}'
            if not (next_grouper in names):
                names[0] = next_grouper
                grouper = next_grouper
                break
    assert names[0] == grouper, 'Unstacking error'

    x.index.rename(names, inplace=True)
    unstacked = [d[1].set_index(d[1].index.get_level_values(1)) for d in list(x.groupby(grouper))]
    if len(unstacked) == 1:
        return unstacked[0]
    else:
        return unstacked
