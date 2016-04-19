import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json
import sklearn.metrics as metrics

from analysis import featurize

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
feature_direc = os.path.join(file_direc, '../features')
model_direc = os.path.join(file_direc, '../models')

ignore_columns = ['total_violations', 'cool', 'funny', 'useful', 'reviews']

if __name__ == "__main__":
    test_df = pd.read_json('%s/test.json' %data_direc, orient='index')
    test_df.loc[test_df.total_violations > 0, 'total_violations'] = 1

    x_test_reviews = featurize.transform_tfidf(test_df['reviews'])

    for c in x_test_reviews.columns.tolist():
        test_df[c] = np.array(x_test_reviews[c])

    x_test = test_df.drop(ignore_columns, axis=1)
    y_test_actual = test_df['total_violations']

    clf = joblib.load('%s/naive_bayes/naive_bayes.pkl' %(model_direc))
    y_test_predicted = clf.predict(x_test)

    mean = np.mean(y_test_predicted == y_test_actual)  
    print "Accuracy: ", mean

    print (metrics.classification_report(y_test_actual, y_test_predicted))
