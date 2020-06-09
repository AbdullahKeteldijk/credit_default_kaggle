import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class EnsembleModel():

    def __init__(self, solver='lbfgs', random_state=42, test_size=0.33, sample_frac=0.8, rounds=100,
                 row_sampling=True, max_iter=100):

        self.solver = solver
        self.random_state = random_state
        self.test_size = test_size
        self.sample_frac = sample_frac
        self.rounds = rounds
        self.max_iter = max_iter
        self.row_sampling = row_sampling

        random.seed(self.random_state)

    def sample_rows(self, X, y, random_state):

        no_sample_rows = int(self.rows * self.sample_frac)
        row_indices = list(range(self.rows))

        X_rows = random.choices(row_indices, k=no_sample_rows)
        X_sample = np.zeros((no_sample_rows, X.shape[1]))
        y_sample = np.zeros((no_sample_rows,))

        for i, row in enumerate(X_rows):
            X_sample[i, :] = X[row, :]
            y_sample[i] = y[row]

        return X_sample, y_sample

    def sample_columns(self, X, y, random_state):

        random.seed(random_state)

        no_sample_cols = int(self.cols * self.sample_frac)
        column_indices = list(range(self.cols))

        X_cols = random.sample(column_indices, no_sample_cols)
        X_sample = X[:, X_cols]

        if self.row_sampling == True:
            X_sample, y_sample = self.sample_rows(X_sample, y, random_state)
        else:
            y_sample = y

        return X_sample, y_sample, X_cols

    def fit(self, X, y):

        self.rows = X.shape[0]
        self.cols = X.shape[1]

        X = X.values
        y = y.values

        self.__clf_dict = {}

        for i in range(self.rounds):
            X_sample, y_sample, X_cols = self.sample_columns(X, y, random_state=i)

            clf = LogisticRegression(solver=self.solver, random_state=self.random_state,
                                     max_iter=self.max_iter)
            clf.fit(X_sample, y_sample)
            self.__clf_dict[i] = [clf, X_cols]

        return None

    def predict(self, X):

        X = X.values
        y_pred_mat = np.zeros((len(X), self.rounds))

        for key, value in self.__clf_dict.items():
            model = value[0]
            cols = value[1]
            y_pred_mat[:, key] = model.predict(X[:, cols])

        y_pred = np.mean(y_pred_mat, axis=1)

        return y_pred, y_pred_mat