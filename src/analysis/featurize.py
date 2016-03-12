import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../../data')
feature_direc = os.path.join(file_direc, '../../features')

def featurize_tfidf(text, max_features=1500):
    vec = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf = vec.fit_transform(text).todense()

    df = pd.DataFrame(data=tfidf, columns=vec.get_feature_names())
    df.to_json('%s/features.json' %feature_direc, orient='columns')
    
    w = open('%s/features_pretty.json' %feature_direc, 'w')
    w.write(json.dumps(json.load(open('%s/features.json' %feature_direc)), indent=4))

    return df

if __name__ == '__main__':

    train_text = pd.read_json('%s/train.json' %data_direc, orient='index')['reviews']

    featurize_tfidf(train_text)


