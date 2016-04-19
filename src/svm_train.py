import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json
from sklearn.linear_model import SGDClassifier

from analysis import featurize

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
feature_direc = os.path.join(file_direc, '../features')
model_direc = os.path.join(file_direc, '../models')

ignore_columns = ['total_violations', 'cool', 'funny', 'useful', 'reviews']

def svm(x_train, y_train):
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    clf.fit(x_train, y_train)
    joblib.dump(clf, '%s/svm/svm.pkl' %(model_direc))

if __name__ == "__main__":
    train_df = pd.read_json('%s/train.json' %data_direc, orient='index')
    train_df.loc[train_df.total_violations > 0, 'total_violations'] = 1

    x_train_reviews = featurize.fit_transform_tfidf(train_df['reviews'])

    for c in x_train_reviews.columns.tolist():
        train_df[c] = np.array(x_train_reviews[c])

    x_train = train_df.drop(ignore_columns, axis=1)
    y_train = train_df['total_violations']

    svm(x_train, y_train)




