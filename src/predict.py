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

def predict(model_name, x_test, y_test):
    clf = joblib.load('%s/%s.pkl' %(model_direc, model_name))
    y_test_predicted = clf.predict(x_test)
    y_test_predicted = np.clip(y_test_predicted, 0, np.inf)
    mse = mean_squared_error(y_test.as_matrix(), y_test_predicted)
    print mse

if __name__ == "__main__":
    print 'Reading test.json'
    test_df = pd.read_json('%s/test.json' %data_direc, orient='index')

    print 'Featurizing test.json'
    x_test = featurize.transform_tfidf(test_df['reviews'])
    y_test = test_df[['*', '**', '***']].astype(np.float64)

    print 'Predicting'
    predict('linear_regression', x_test, y_test)






