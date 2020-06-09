import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class EnsembleModel():

    '''
    This Ensemble model is functions similar to a random forest. It stacks multiple Logistic Regression models.
    The models are all trained on different versions of the data. These versions are created by random sampling columns
    (without replacement) and rows (with replacement). It ensured that the Ensemble model does not overfit while at
    the same time it performs better than any single model.
    '''

    def __init__(self, solver='lbfgs', random_state=42, test_size=0.33, sample_frac=0.8, rounds=100,
                 row_sampling=True, max_iter=100):

        ''''
        :param solver: string - See scikit-learn documentation for all posible solvers for the Logistic Regression
        :param random_state: int - A number that is used as a random seed to ensure the model performs exactly the same
            each it is run with a given set of hyperparameters
        :param test_size: float - The fraction of the dataframe thest should be used as a test set
        :param sample_frac: float - The fraction of the rows and columns that should be sampled for the ensemble model
        :param rounds: int - Number of sampling rounds for the Ensemble model
        :param row_sampling: boolean - Indicating whether the rows should be sampled in addition to the columns
        :param max_iter: int - Maximum number of iterations for the Logistic Regression models
        '''

        self.solver = solver
        self.random_state = random_state
        self.test_size = test_size
        self.sample_frac = sample_frac
        self.rounds = rounds
        self.max_iter = max_iter
        self.row_sampling = row_sampling

        random.seed(self.random_state)

    def sample_rows(self, X, y, random_state):
        '''
        This function samples the rows (with replacement) for the Ensemble model.

        :param X: numpy array - Matrix containing all the features
        :param y: numpy array - Vector containing the target variable
        :param random_state: int - A number that is used as a random seed to ensure the model performs exactly the same
        each it is run with a given set of hyperparameters

        :return X_sample: numpy array - Matrix containing all the sampled rows of the features
        :return y_sample: numpy array - Vector containing all the sampled rows of the target variable
        '''

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
        '''
        This function samples the columns (without replacement) for the Ensemble model.

        :param X: numpy array - Matrix containing all the features
        :param y: numpy array - Vector containing the target variable
        :param random_state: int - A number that is used as a random seed to ensure the model performs exactly the same
        each it is run with a given set of hyperparameters

        :return X_sample: numpy array - Matrix containing all the sampled columns (and rows) of the features
        :return y_sample: numpy array - Vector containing all the (sampled) rows of the target variable
        :return X_cols: list - A list containing the column numbers of the samples column to keep track of the sampling
        '''

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
        '''
        This function executes all the previous functions. It runs the Logistic Regression models as specified at
        the beginning. All models and the sampled columns are then stored in a dictionary in order to be used for the
        prediction.

        :param X: numpy array - Matrix containing all the features
        :param y: numpy array - Vector containing the target variable
        '''
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
        '''
        This function generates a prediction for each trained model. These predictions are then stored in a matrix
        and the mean is calculated. One can either use the mean prediction or the prediction matrix to train an
        additional model on top in order to increase the performance of the model even more.

        :param X: numpy array - Matrix containing all the features

        :return y_pred: numpy array - Vector containing the mean predicted probabilities of all models
        :return y_pred_mat: numpy array - Matrix containing the predictions of all the models in the Ensemble model
        '''


        X = X.values
        y_pred_mat = np.zeros((len(X), self.rounds))

        for key, value in self.__clf_dict.items():
            model = value[0]
            cols = value[1]
            y_pred_mat[:, key] = model.predict(X[:, cols])

        y_pred = np.mean(y_pred_mat, axis=1)

        return y_pred, y_pred_mat