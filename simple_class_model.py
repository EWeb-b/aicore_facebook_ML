import numpy as np
import pandas as pd

from matplotlib.image import imread
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class SimpleClassModel():
    def __init__(self, data):
        self.df = data

    def get_numpy_representations(self):
        """Converts the columns into numpy arrays and also reshapes the image arrays
        so that they're compatible with the model.

        Returns:
            X: the numpy array of images, reshaped.
            y, the numpy array of categories.
        """
        X_stack = np.stack(self.df['img_arr'].values)
        X = X_stack.reshape(self.df.shape[0], 128*128*3)

        y = np.stack(self.df['category'].values)

        return X, y

    def run_classification(self):
        """Runs the classification and prints the accuracy report to the terminal.
        """
        X, y = self.get_numpy_representations()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        print("accuracy: ", accuracy_score(y_test, prediction))
        print("report: ", classification_report(y_test, prediction))


if __name__ == "__main__":
    img_cat_data = pd.read_pickle('data/cl_img_pickle.pkl')
    classifier = SimpleClassModel(img_cat_data)
    classifier.run_classification()

