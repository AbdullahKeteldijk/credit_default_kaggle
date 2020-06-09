import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from src.data_cleaner import DataCleaner
from src.ensemble_model import EnsembleModel


class ModelPipeline():

    def __init__(self, df, test_size=0.33, random_state=42, row_sampling=True,
                 rounds=250, max_iter=100, simple_model=False, mean_model=False,
                 cutoff=0, model_name=None, solver='lbfgs'):

        self.df = df
        self.test_size = test_size
        self.random_state = random_state
        self.row_sampling = row_sampling
        self.rounds = rounds
        self.max_iter = max_iter
        self.simple_model = simple_model
        self.mean_model = mean_model
        self.cutoff = cutoff
        self.model_name = model_name
        self.solver = solver

    def probabilities_to_int(self, y_pred):

        for i in range(len(y_pred)):
            if y_pred[i] > self.cutoff:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred.astype(int)

    @ignore_warnings(category=ConvergenceWarning)
    def fit_transform(self):

        self.df = DataCleaner(self.df).clean()

        X = self.df.drop(['BAD'], axis=1)
        y = self.df['BAD']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size,
                                                                                random_state=self.random_state)

        if self.simple_model:
            self.simp = LogisticRegression(random_state=42)
            self.simp.fit(self.X_train, self.y_train)
            self.prediction = self.simp.predict(self.X_test)

        else:
            self.ens = EnsembleModel(row_sampling=self.row_sampling, random_state=self.random_state,
                                     rounds=self.rounds, max_iter=self.max_iter, solver=self.solver)
            self.ens.fit(self.X_train, self.y_train)
            y_pred, y_pred_mat = self.ens.predict(self.X_test)

            if self.mean_model:
                self.prediction = self.probabilities_to_int(self.y_pred)

            else:
                self.lr_pred = LogisticRegression(random_state=42, solver=self.solver)
                self.lr_pred.fit(y_pred_mat, self.y_test)
                self.prediction = self.lr_pred.predict(y_pred_mat)

    def generate_model_name(self):

        if self.simple_model:
            self.model_name = "Simple Logit model"
        elif self.mean_model:
            self.model_name = "Ensemble model with mean prediction"
        else:
            self.model_name = "Ensemble model with modelled prediction"

    def print_model_performance(self):

        if self.model_name == None:
            self.generate_model_name()

        print('Model:', self.model_name)
        print('Accuracy score:', accuracy_score(self.y_test, self.prediction))
        print('F1 score:', f1_score(self.y_test, self.prediction))
        print('Precision score:', precision_score(self.y_test, self.prediction))
        print('Recall score:', recall_score(self.y_test, self.prediction))