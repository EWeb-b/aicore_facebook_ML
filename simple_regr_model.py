import clean_tabular_data

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df = clean_tabular_data.clean_tab_data()

x_columns = ['product_name', 'product_description', 'location']
X = df[x_columns]

target_column = 'price'
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)