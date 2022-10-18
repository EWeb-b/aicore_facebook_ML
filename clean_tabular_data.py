import pandas as pd
import numpy as np

from pandas import DataFrame


def clean_tab_data() -> DataFrame:
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("data/Products.csv", lineterminator="\n")
    # print(df["product_name"])

    # Create a copy to work on.
    df_copy = df.copy()

    # Strip the pound sign, remove commas from price column and convert it into a float type.
    df_copy['price'] = df_copy['price'] \
        .apply(lambda x: x.replace('£', '').replace(',','')) \
        .astype('float64')

    # Change the type of the 'category' column to be of type 'category'.
    df_copy['category'] = df_copy['category'].astype('category')

    # Convert all the types of the columns to be best possible types.
    df_copy = df_copy.convert_dtypes(convert_integer=False, convert_floating=False)

    return df_copy