import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
df = pd.read_csv("data/Products.csv", lineterminator="\n")
# print(df["product_name"])

# Create a copy to work on.
df_copy = df.copy()

# Strip the pound sign from price column and convert it into a float type.
df_copy['price'] = df_copy['price'].str.strip('£')
df_copy['price'] = df_copy['price'].str.replace(",", "")
df_copy['price'] = df_copy['price'].astype('float64')
