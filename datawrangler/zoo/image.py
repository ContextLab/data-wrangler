import PIL
import six
import os
import numpy as np
from matplotlib import pyplot as plt

from .array import wrangle_array
from ..util import array_like, btwn


def get_image(img):
    """
    Return an Array containing the image data reflected in the given array-like object, URL, or filename

    Parameters
    ----------
    :param img: an array-like object, URL, or filename

    Returns
    -------
    :return: an Array containing the image data (if img is a valid image), or None if img is not a valid image.
    """
    def valid_image_values(x):
        if not (hasattr(x, 'dtype') and hasattr(x, 'ndim')):
            return False

        dtype = str(x.dtype)
        if 'int' in dtype:
            return btwn(x, 0, 255)
        elif 'float' in dtype:
            return btwn(x, 0.0, 1.0)
        else:
            return False

    if (type(img) in six.string_types) and os.path.exists(img):
        try:
            return plt.imread(img)  # also handles remote images
        except PIL.UnidentifiedImageError:
            return None

    if array_like(img):
        if valid_image_values(img):
            if img.ndim == 2:
                return img
            elif img.ndim == 3 and img.shape[2] in [3, 4]:
                return img
    return None
