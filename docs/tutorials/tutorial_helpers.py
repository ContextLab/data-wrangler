import os


def parent_directory(d, n=0):
    """
    Go up n levels in the directory tree

    Parameters
    ----------
    :param d: reference file
    :param n: number of levels to go up in the directory tree

    Returns
    -------
    :return: nth parent of the reference file
    """

    if n == 0:
        return os.path.split(d)[0]
    else:
        return parent_directory(parent_directory(d), n-1)


resources = os.path.join(parent_directory(os.path.dirname(__file__), 1), 'tests', 'resources')

data_file = os.path.join(resources, 'testdata.csv')
image_file = os.path.join(resources, 'wrangler.jpg')
text_file = os.path.join(resources, 'home_on_the_range.txt')
