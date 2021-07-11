import six
from matplotlib import pyplot as plt

from .array import wrangle_array
from ..util import array_like
from ..io.io import get_extension
from ..util.helpers import btwn


def get_image(img):
    def valid_image_values(x):
        dtype = str(x.dtype)
        if 'int' in dtype:
            return btwn(x, 0, 255)
        elif 'float' in dtype:
            return btwn(x, 0.0, 1.0)
        else:
            return False

    if type(img) in six.string_types:
        ext = get_extension(img)
        if ext.lower in plt.gcf().canvas.get_supported_filetypes().keys():
            return plt.imread(img)  # also handles remote images

    if array_like(img):
        if valid_image_values(img):
            if img.ndim == 1:
                return img
            elif img.ndim == 3 and img.shape[2] in [3, 4]:
                return img
    return None


# noinspection PyUnusedLocal
def is_image(data):
    img = get_image(data)
    return img is not None


# noinspection PyUnusedLocal
def wrangle_image(data, **kwargs):
    img = get_image(data)

    if img is not None:
        return wrangle_array(np.hstack([img[:, :, i] for i in range(img.shape[2])]))
