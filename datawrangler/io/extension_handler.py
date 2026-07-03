import os


def get_extension(fname):
    """
    Return the (lowercase) extension of a file, or return "dw" if the extension could not be determined.

    Any URL query string (``?...``) or fragment (``#...``) is stripped before the extension is
    extracted, so remote URLs such as ``https://.../data.npz?dl=1`` (Dropbox, Google Drive, etc.)
    resolve to their true extension (``npz``) rather than ``npz?dl=1``.  Without this, cached copies
    of query-string URLs were saved under names like ``<hash>.npz?dl=1`` and could not be re-read
    (``Unknown datatype: npz?dl=1``), causing repeated downloads.

    Parameters
    ----------
    :param fname: the filename or URL, represented as a string

    Returns
    -------
    :return: The extension, represented as a lowercase string.
    """
    # Drop the URL query string / fragment before locating the extension.
    fname = fname.split('?', 1)[0].split('#', 1)[0]
    _, f = os.path.split(fname)
    if '.' in f:
        return f[f.rfind('.') + 1:].lower()
    return 'dw'
