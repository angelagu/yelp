import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import json
from sklearn.naive_bayes import MultinomialNB
from analysis import featurize
from sklearn.metrics import mean_squared_error

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
feature_direc = os.path.join(file_direc, '../features')
model_direc = os.path.join(file_direc, '../models')

if __name__ == "__main__":
    test_df = pd.read_json('%s/test.json' %data_direc, orient='index')

    x_test = featurize.transform_tfidf(test_df['reviews'])

    model_name = 'naive_bayes'

    i = 0
    clf_0 = joblib.load('%s/NB/%d/%s.pkl' %(model_direc, i, model_name))
    y_test_predicted_0 = clf_0.predict(x_test)
    y_test_0 = test_df[['*']].astype(np.float64)
    mse_0 = mean_squared_error(y_test_0.as_matrix(), y_test_predicted_0)

    i = 1
    clf_1 = joblib.load('%s/NB/%d/%s.pkl' %(model_direc, i, model_name))
    y_test_predicted_1 = clf_1.predict(x_test)
    y_test_1 = test_df[['**']].astype(np.float64)
    mse_1 = mean_squared_error(y_test_1.as_matrix(), y_test_predicted_1)

    i = 2
    clf_2 = joblib.load('%s/NB/%d/%s.pkl' %(model_direc, i, model_name))
    y_test_predicted_2 = clf_2.predict(x_test)
    y_test_2 = test_df[['***']].astype(np.float64)
    mse_2 = mean_squared_error(y_test_2.as_matrix(), y_test_predicted_2)

    mse = (mse_0 + mse_1 + mse_2) / 3

    print mse





