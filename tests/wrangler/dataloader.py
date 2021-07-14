import os
import pandas as pd

resources = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'resources')
data_file = os.path.join(resources, 'testdata.csv')
img_file = os.path.join(resources, 'wrangler.jpg')
text_file = os.path.join(resources, 'home_on_the_range.txt')

data = pd.read_csv(data_file, index_col=0)