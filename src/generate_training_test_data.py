import pandas as pd
import numpy as np
import os
from sklearn.cross_validation import train_test_split
import json

file_direc = os.path.dirname(__file__)
data_direc = os.path.join(file_direc, '../data')
test_size = 0.20

def get_flattened_reviews(reviews_df, label_df):
    reviews_dictionary = {}

    N = len(label_df)

    for i, (pid, row) in enumerate(label_df.iterrows()):
        pre_inspection_mask = (reviews_df.restaurant_id == row.restaurant_id)
        
        pre_inspection_reviews = reviews_df[pre_inspection_mask].text.tolist()

        all_text = ' '.join(pre_inspection_reviews)

        reviews_dictionary[pid] = all_text

        if i % 2500 == 0:
            print '{} out of {}'.format(i, N)

    return pd.Series(reviews_dictionary)[label_df.index]

def map_reviews_to_restaurant_ids(id_map, reviews):
    id_dict = {}

    for i, row in id_map.iterrows():
        boston_id = row["restaurant_id"]

        non_null_mask = ~pd.isnull(row.ix[1:])
        yelp_ids = row[1:][non_null_mask].values

        for yelp_id in yelp_ids:
            id_dict[yelp_id] = boston_id

    map_to_boston_ids = lambda yelp_id: id_dict[yelp_id] if yelp_id in id_dict else np.nan
    reviews.business_id = reviews.business_id.map(map_to_boston_ids)

    reviews.columns = ["restaurant_id", "date", "stars", "text"]    
    reviews = reviews[pd.notnull(reviews.restaurant_id)]

    return reviews

if __name__ == '__main__':

    id_map = pd.read_csv("%s/restaurant_ids_to_yelp_ids.csv" %data_direc)

    with open("%s/yelp_academic_dataset_review.json" %data_direc, 'r') as review_file:
        review_json = '[' + ','.join(review_file.readlines()) + ']'

    reviews_df = pd.read_json(review_json)
    reviews_df.drop(['review_id', 'type', 'user_id', 'votes'], 
             inplace=True, 
             axis=1)

    reviews_df = map_reviews_to_restaurant_ids(id_map, reviews_df)

    label_df = pd.read_csv("%s/AllViolations.csv" %data_direc, index_col=0)
    flattened_reviews = get_flattened_reviews(reviews_df, label_df)

    labels = label_df[['*', '**', '***']].astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(flattened_reviews, labels, test_size=test_size, random_state=42)

    print 'Length of training set: ', len(X_train)
    print 'Length of test set: ', len(X_test)

    X_train_df = X_train.to_frame(name='reviews')
    X_test_df = X_test.to_frame(name='reviews')

    train_df = pd.merge(X_train_df, y_train, how='inner', left_index=True, right_index=True)
    test_df = pd.merge(X_test_df, y_test, how='inner', left_index=True, right_index=True)

    train_df.reset_index(drop=True).to_json('%s/train.json' %data_direc, orient='index')
    test_df.reset_index(drop=True).to_json('%s/test.json' %data_direc, orient='index')

    w = open('%s/train_pretty.json'  %data_direc, 'w')
    w.write(json.dumps(json.load(open('%s/train.json' %data_direc)), indent=4))

    w = open('%s/test_pretty.json' %data_direc, 'w')
    w.write(json.dumps(json.load(open('%s/test.json' %data_direc)), indent=4))



