import os


def get_extension(fname):
    """
    Return the (lowercase) extension of a file, or return "dw" if the extension could not be determined.

    Parameters
    ----------
    :param fname: the filename, represented as a string

    Returns
    -------
    :return: The extension, represented as a lowercase string.
    """
    _, f = os.path.split(fname)
    if '.' in f:
        return f[f.rfind('.') + 1:].lower()
    return 'dw'
