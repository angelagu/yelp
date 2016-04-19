import os
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import featurize

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../../data')
plots_direc = os.path.join(file_direc, '../plots')

ignore_columns = ['total_violations', 'cool', 'funny', 'useful', 'reviews']

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

if __name__ == "__main__":
    train_df = pd.read_json('%s/train.json' %data_direc, orient='index')
    train_df.loc[train_df.total_violations > 0, 'total_violations'] = 1

    x_train_reviews = featurize.fit_transform_tfidf(train_df['reviews'])

    for c in x_train_reviews.columns.tolist():
        train_df[c] = np.array(x_train_reviews[c])

    x_train = train_df.drop(ignore_columns, axis=1)
    y_train = train_df['total_violations']

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x_train, y_train)

    plot_pca(X_pca, y_train)




