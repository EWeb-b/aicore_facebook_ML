import clean_tabular_data

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# Set the data.
df = clean_tabular_data.clean_tab_data()

# Combine and set the features.
X = pd.DataFrame()
X['combined_features'] = df['product_name'] + df['product_description'] + df['location']

# Set the target.
y = df['price']

tfidf_vectorizer = TfidfVectorizer()
vec = tfidf_vectorizer.fit_transform(X['combined_features'])

lr = LinearRegression()
lr.fit(vec, y)

error = mean_squared_error(y, lr.predict(vec))
print(error)