import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
feature_direc = os.path.join(file_direc, '../features')
model_direc = os.path.join(file_direc, '../models')

def get_top_features(features, model, level, limit, bottom=False):
    """ Get the top (most likely to see violations) and bottom (least
        likely to see violations) features for a given model.
        
        :param features: an array of the feature names
        :param model: a fitted linear regression model
        :param level: 0, 1, 2 for *, **, *** violation levels
        :param limit: how many features to return
        :param worst: if we want the bottom features rather than the top 
    """
    # sort order for the coefficients
    sorted_coeffs = np.argsort(model.coef_[i])
    
    if bottom:
        # get the features at the end of the sorted list
        return features[sorted_coeffs[-1 * limit:]]
    else:
        # get the features at the beginning of the sorted list
        return features[sorted_coeffs[:limit]]

def naive_bayes(x_train, y_train, i):
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    joblib.dump(clf, '%s/NB/%d/naive_bayes.pkl' %(model_direc, i))

if __name__ == "__main__":

    x_train = pd.read_json('%s/features.json' %feature_direc, orient='columns')
    x_train = x_train.sort_index()

    train_df = pd.read_json('%s/train.json' %data_direc, orient='index')

    print(x_train)

    i = 0
    y_train_0 = train_df[['*']].astype(np.float64)
    naive_bayes(x_train, y_train_0, i)

    i = 1
    y_train_1 = train_df[['**']].astype(np.float64)
    naive_bayes(x_train, y_train_1, i)

    i = 2
    y_train_2 = train_df[['***']].astype(np.float64)
    naive_bayes(x_train, y_train_2, i)




