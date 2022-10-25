import pandas as pd

img_cat_data = pd.read_pickle('data/cl_img_pickle.pkl')

#df = img_cat_data.groupby(['category'])['category'].count()

img_cat_data.head(50)