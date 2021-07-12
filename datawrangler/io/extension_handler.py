import os


def get_extension(fname):
    _, f = os.path.split(fname)
    if '.' in f:
        return f[f.rfind('.') + 1:].lower()
    return None
