import os
import pandas as pd
import numpy as np
import numpy.ma as ma

from IPython.display import Image 
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.externals import joblib
import json
import pydot

from analysis import featurize

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
feature_direc = os.path.join(file_direc, '../features')
model_direc = os.path.join(file_direc, '../models')
plots_direc = os.path.join(file_direc, 'plots')

ignore_columns = ['total_violations', 'reviews', 'restaurant_id']

def svm(x_train, y_train):
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    clf.fit(x_train, y_train)
    joblib.dump(clf, '%s/svm/svm.pkl' %(model_direc))

def logistic_regression(x_train, y_train):
    clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    clf.fit(x_train, y_train)
    joblib.dump(clf, '%s/logistic_regression/logistic_regression.pkl' %(model_direc))

def pca(x_train, y_train):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x_train, y_train)
    plot_pca(X_pca, y_train)

def naive_bayes(x_train, y_train):
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    joblib.dump(clf, '%s/naive_bayes/naive_bayes.pkl' %(model_direc))

def decision_tree(x_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(x_train[:1500], y_train[:1500])
    joblib.dump(clf, '%s/decision_tree/decision_tree.pkl' %(model_direc))

    visualize_tree(clf, x_train.columns)

def visualize_tree(tree, feature_names):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=feature_names)

    graph = pydot.graph_from_dot_file('dt.dot')
    graph.write_png('%s/decision_tree.png' %plots_direc)

def plot_pca(X, y):
    y = np.array([(i, i) for i in y])
    plt.figure()
    for c, i, target_name in zip(["r", "g"], [0, 1], ["Has violations", "No violations"]):
        mx = ma.masked_array(X, mask=y==i)
        plt.scatter(mx[:, 0], mx[:, 1], c=c, label=target_name)

    plt.axis('tight')
    lgd = plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.9), fancybox=True, shadow=True)
    plt.title('PCA of Yelp Reviews')
    plt.savefig('%s/pca.png' %(plots_direc), bbox_extra_artists=(lgd,), bbox_inches='tight')

def predict(model_name):
    test_df = pd.read_json('%s/test.json' %data_direc, orient='index')
    test_df.loc[test_df.total_violations > 0, 'total_violations'] = 1

    x_test_reviews = featurize.transform_tfidf(test_df['reviews'])

    for c in x_test_reviews.columns.tolist():
        test_df[c] = np.array(x_test_reviews[c])

    x_test = test_df.drop(ignore_columns, axis=1)
    y_test_actual = test_df['total_violations']

    clf = joblib.load('%s/%s/%s.pkl' %(model_direc, model_name, model_name))
    y_test_predicted = clf.predict(x_test)

    print model_name

    mean = np.mean(y_test_predicted == y_test_actual)  
    print "Accuracy: ", mean

    print (metrics.classification_report(y_test_actual, y_test_predicted))

def train_all_models():
    train_df = pd.read_json('%s/train.json' %data_direc, orient='index')
    train_df.loc[train_df.total_violations > 0, 'total_violations'] = 1

    x_train_reviews = featurize.fit_transform_tfidf(train_df['reviews'], max_features=10)

    for c in x_train_reviews.columns.tolist():
        train_df[c] = np.array(x_train_reviews[c])

    x_train = train_df.drop(ignore_columns, axis=1)
    y_train = train_df['total_violations']

    svm(x_train, y_train)
    naive_bayes(x_train, y_train)
    logistic_regression(x_train, y_train)
    decision_tree(x_train, y_train)
    pca(x_train, y_train)

if __name__ == "__main__":
    train_all_models()
    predict('svm')
    predict('naive_bayes')
    predict('logistic_regression')
    predict('decision_tree')


