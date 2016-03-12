import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
feature_direc = os.path.join(file_direc, '../features')

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
    
def featurize(train_text, max_features=1500):
    vec = TfidfVectorizer(stop_words='english', max_features=max_features)
    train_tfidf = vec.fit_transform(train_text)

    df = pd.DataFrame(data=train_tfidf.todense(), columns=vec.get_feature_names())
    df.to_json('%s/features.json' %feature_direc, orient='index')


if __name__ == '__main__':

    train_text = pd.read_json('%s/train.json' %data_direc, orient='index')['reviews']

    featurize(train_text)

