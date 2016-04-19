import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json
from sklearn.naive_bayes import MultinomialNB

from analysis import featurize

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
feature_direc = os.path.join(file_direc, '../features')
model_direc = os.path.join(file_direc, '../models')

ignore_columns = ['total_violations', 'cool', 'funny', 'useful', 'reviews']

def naive_bayes(x_train, y_train):
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    joblib.dump(clf, '%s/naive_bayes/naive_bayes.pkl' %(model_direc))

if __name__ == "__main__":
    train_df = pd.read_json('%s/train.json' %data_direc, orient='index')
    train_df.loc[train_df.total_violations > 0, 'total_violations'] = 1

    x_train_reviews = featurize.fit_transform_tfidf(train_df['reviews'])

    for c in x_train_reviews.columns.tolist():
        train_df[c] = np.array(x_train_reviews[c])

    x_train = train_df.drop(ignore_columns, axis=1)
    y_train = train_df['total_violations']

    naive_bayes(x_train, y_train)




