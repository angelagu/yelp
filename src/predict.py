import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import json
from sklearn import linear_model
from analysis import featurize
from sklearn.metrics import mean_squared_error

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
feature_direc = os.path.join(file_direc, '../features')
model_direc = os.path.join(file_direc, '../models')

if __name__ == "__main__":
    test_df = pd.read_json('%s/test.json' %data_direc, orient='index')

    x_test = featurize.featurize_tfidf(test_df['reviews'], save=False)
    y_test = test_df[['*', '**', '***']].astype(np.float64)

    model_name = 'logistic_regression'
    clf = joblib.load('%s/%s.pkl' %(model_direc, model_name))

    y_test_predicted = clf.predict(x_test)

    mse = mean_squared_error(y_test.as_matrix(), y_test_predicted)

    print mse





