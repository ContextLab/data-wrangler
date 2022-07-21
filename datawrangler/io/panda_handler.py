import six
import warnings
import pandas as pd

from .extension_handler import get_extension


def load_dataframe(x, extension=None, debug=False, **kwargs):
    """
    An internal function for handling the variety of data types supported by Pandas

    Parameters
    ----------
    :param x: a string (filename or URL) or DataFrame
    :param extension: optional argument for specifying the filetype (e.g., 'csv', 'xls', etc.).  If no extension is specified,
      by default the extension will be inferred automatically from the filename.  The following data types are
      supported: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
    :param debug: internal flag used for debugging (prints a warning message if filetype cannot be inferred automatically);
      default: False.
    :param kwargs: any additional keyword arguments are passed directly to the relevant Pandas function for reading in the
      inferred data type.

    Returns
    -------
    :return: a DataFrame containing the file's contents
    """
    if type(x) in six.string_types:
        if extension is None:
            extension = get_extension(x)

        # built-in pandas parsers support both local and remote loading
        if extension == 'csv':
            return pd.read_csv(x, **kwargs)
        elif extension in ['xls', 'xlsx']:
            return pd.read_excel(x, **kwargs)
        elif extension == 'json':
            return pd.read_json(x, **kwargs)
        elif extension == 'html':
            return pd.read_html(x, **kwargs)
        elif extension == 'xml':
            return pd.read_xml(x, **kwargs)
        elif extension == 'hdf':
            return pd.read_hdf(x, **kwargs)
        elif extension == 'feather':
            return pd.read_feather(x, **kwargs)
        elif extension == 'parquet':
            return pd.read_parquet(x, **kwargs)
        elif extension == 'orc':
            return pd.read_orc(x, **kwargs)
        elif extension == 'sas':
            return pd.read_sas(x, **kwargs)
        elif extension == 'spss':
            return pd.read_spss(x, **kwargs)
        elif extension == 'sql':
            return pd.read_sql(x, **kwargs)
        elif extension == 'gbq':
            return pd.read_gbq(x, **kwargs)
        elif extension == 'stata':
            return pd.read_stata(x, **kwargs)
        elif extension == 'pkl':
            return pd.read_pickle(x, **kwargs)
        else:
            if debug:
                warnings.warn(f'cannot determine filetype: {x}')
            return None
    elif all([d in type(x).__module__.lower() for d in ['pandas', 'frame']]):
        return x
    else:
        return None
