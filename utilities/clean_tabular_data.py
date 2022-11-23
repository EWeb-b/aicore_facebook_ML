import pandas as pd
import numpy as np

from pandas import DataFrame


def clean_tab_data(filepath: str = '/home/edwardwebb/Documents/aicore/aicore_facebook_ML/data/Products.csv', labels_level: int = 0) -> DataFrame:
    df = pd.read_csv(filepath, lineterminator="\n")

    # Create a copy to work on.
    df_copy = df.copy()

    # Strip the pound sign, remove commas from price column and convert it into a float type.
    df_copy['price'] = df_copy['price'] \
        .apply(lambda x: x.replace('£', '').replace(',','')) \
        .astype('float64')

    # Change the type of the 'category' column to be of type 'category'.
    df_copy['category'] = df_copy['category'].apply(lambda x: x.split("/")[labels_level].strip())
    df_copy['category'] = df_copy['category'].astype('category')

    df.drop(['price'], axis=1, inplace=True)

    # Convert all the types of the columns to be best possible types.
    df_copy = df_copy.convert_dtypes(convert_integer=False, convert_floating=False)

    return df_copy

if __name__ == "__main__":
    df = clean_tab_data()
    print(df.category.cat.categories[3])
    print(df.category.cat.categories)
    print(type(df.category.cat.categories))
    print(df.category.cat.categories.get_loc('Sports, Leisure & Travel'))
    print(len(df.category.cat.categories))

    # print(type(df.category.cat.codes))