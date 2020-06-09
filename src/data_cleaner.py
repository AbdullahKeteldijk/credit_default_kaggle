import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


class DataCleaner:

    def __init__(self, df, predict_nan=True):

        self.df = df.copy()
        self.predict_nan = predict_nan

    def fill_na_model(self, df_train, target):

        X = df_train.drop(['BAD', 'DEROG', 'DELINQ'], axis=1)
        y = df_train[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = GradientBoostingRegressor(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        return clf

    def predict_na(self, target, clf=None):

        df_train = self.df[~self.df[target].isna()]
        df_test = self.df[self.df[target].isna()]
        df_test_ = df_test.drop(columns=['BAD', 'DEROG', 'DELINQ'], axis=1)

        if clf == None:
            clf = self.fill_na_model(df_train, target)

        y_pred = clf.predict(df_test_)
        df_test[target] = y_pred
        df_new = pd.concat([df_train, df_test])

        return df_new, clf

    def dummy_replace(self, column, prefix=None):

        df_dummy = pd.get_dummies(self.df[column], prefix=prefix)
        self.df = pd.concat([self.df, df_dummy], axis=1)
        self.df = self.df.drop(columns=[column], axis=1)

    def fill_nan(self):

        self.df['MORTDUE'] = self.df['MORTDUE'].fillna(self.df['MORTDUE'].median())
        self.df['VALUE'] = self.df['VALUE'].fillna(self.df['VALUE'].median())
        self.df['REASON'] = self.df['REASON'].fillna('Unknown')
        self.df['JOB'] = self.df['JOB'].fillna('Unknown')
        self.df['YOJ'] = self.df['YOJ'].fillna(self.df['YOJ'].median())
        self.df['CLAGE'] = self.df['CLAGE'].fillna(self.df['CLAGE'].median())
        self.df['NINQ'] = self.df['NINQ'].fillna(self.df['NINQ'].median())
        self.df['CLNO'] = self.df['CLNO'].fillna(self.df['CLNO'].median())
        self.df['DEBTINC'] = self.df['DEBTINC'].fillna(self.df['DEBTINC'].median())

        if self.predict_nan == False:
            self.df['DEROG'] = self.df['DEROG'].fillna(self.df['DEROG'].mean())
            self.df['DELINQ'] = self.df['DELINQ'].fillna(self.df['DELINQ'].mean())

    def feature_engineering(self):

        self.df['LOAN'] = np.log(self.df['LOAN'])
        self.df['MORTDUE'] = np.log(self.df['MORTDUE'])
        self.df['VALUE'] = np.log(self.df['VALUE'])
        self.df['YOJ'] = np.sqrt(self.df['YOJ'])

    def clean(self, clf_derog=None, clf_delinq=None):

        self.fill_nan()
        self.feature_engineering()

        self.dummy_replace('REASON', prefix='Reason')
        self.dummy_replace('JOB', prefix='Job')

        if self.predict_nan:
            self.df, self.clf_derog = self.predict_na('DEROG', clf=clf_derog)
            self.df, self.clf_delinq = self.predict_na('DELINQ', clf=clf_delinq)

        return self.df