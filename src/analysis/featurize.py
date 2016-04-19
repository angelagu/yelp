import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as t
from sklearn.externals import joblib
import Stemmer

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../../data')
feature_direc = os.path.join(file_direc, '../../features')

english_stemmer = Stemmer.Stemmer('en')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: english_stemmer.stemWords(analyzer(doc))

def transform_tfidf(text):
    vec = joblib.load('%s/tfidf_vector.pk' %feature_direc) 
    tfidf = vec.transform(text).todense()
    df = pd.DataFrame(data=tfidf, columns=vec.get_feature_names())
    return df

def fit_transform_tfidf(text, max_features=1500):
    print 'Generating features...'

    vec = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf = vec.fit_transform(text).todense()

    indices = np.argsort(vec.idf_)[::-1]
    features = vec.get_feature_names()
    top_n = 100
    top_features = [features[i] for i in indices[:top_n]]
    print top_features

    joblib.dump(vec, '%s/tfidf_vector.pk' %feature_direc) 

    df = pd.DataFrame(data=tfidf, columns=vec.get_feature_names())

    print 'Saving features'
    df.to_json('%s/features.json' %feature_direc, orient='columns')
    w = open('%s/features_pretty.json' %feature_direc, 'w')
    w.write(json.dumps(json.load(open('%s/features.json' %feature_direc)), indent=4))

    return df

if __name__ == '__main__':
    print 'Reading train.json'
    train_text = pd.read_json('%s/train.json' %data_direc, orient='index')['reviews']

    fit_transform_tfidf(train_text, max_features=1000)


