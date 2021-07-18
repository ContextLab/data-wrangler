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
        if not hasattr(x, 'dtype'):
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
            if img.ndim == 1:
                return img
            elif img.ndim == 3 and img.shape[2] in [3, 4]:
                return img
    return None


# noinspection PyUnusedLocal
def is_image(data):
    """
    Test whether an object is a valid image, or a URL or filename that points to a valid image.

    Parameters
    ----------
    :param data: the to-be-tested object

    Returns
    -------
    :return: True if the object is an image and False otherwise
    """
    img = get_image(data)
    return img is not None


# noinspection PyUnusedLocal
def wrangle_image(data, **kwargs):
    """
    Turn an image into a DataFrame by horizontally concatenating its color and/or alpha channels.

    Parameters
    ----------
    :param data: the to-be-wrangled image
    :param kwargs: any keyword objects are passed on to dataframe.zoo.wrangle_array

    Returns
    -------
    :return: a DataFrame containing the wrangled image data
    """
    img = get_image(data)

    if img is not None:
        return wrangle_array(img, **kwargs)
