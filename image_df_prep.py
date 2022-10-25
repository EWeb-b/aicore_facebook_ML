import pandas as pd
import numpy as np
from PIL import Image

# Load in the csv files.
pdf = pd.read_csv("data/Products.csv", lineterminator="\n")
idf = pd.read_csv("data/Images.csv")

# Merge the dataframes from the csv files. They are merged on the product_id, which both csv files contain. This means
# that the images are paired with the relevant category.
result = pd.merge(pdf, idf, left_on='id', right_on='product_id')
df = result.copy()
df.drop(['id_x', 'product_name', 'product_description', 'price', 'location', 'product_id', 'Unnamed: 0_x', 'Unnamed: 0_y'], axis=1, inplace=True)

# Convert the categories into numerical values.
df['category'] = df['category'].apply(lambda x: x.split("/")[0].strip())
df['category'] = df['category'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda c: c.cat.codes)

# Convert the images into their numpy array representation.
df['id_y'] = df['id_y'].apply(lambda z: np.array(Image.open('data/cleaned_images/' + z + '.jpg')))

# Rename the column containing the image arrays for ease of use.
df = df.rename({'id_y': 'img_arr'}, axis = 1)

# Save the dataframe as a pickle file, ready for classification model.
df.to_pickle('data/cl_img_pickle.pkl')
