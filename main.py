# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from __future__ import division
from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
from random import shuffle
import numpy as np
import json
import parsers as ps
import classifiers as cf

CATEGORIES = [
    'Musical_Instruments',
    'Patio_Lawn_and_Garden',
    'Amazon_Instant_Video'
]

def load_reviews(path, percent):
    json_reviews = open(path).read().split('\n')[:-1]
    reviews = []
    for json_review in json_reviews:
        raw_review = json.loads(json_review)
        review = [
        	ps.idToNum(raw_review['reviewerID']),
        	ps.idToNum(raw_review['asin']),
        	ps.helpfulToNum(raw_review['helpful']),
        	ps.overallToNum(raw_review['overall']),
        	raw_review['unixReviewTime'],
        	raw_review['summary'],
        	raw_review['reviewText']
    	]
        reviews.append(review)

    shuffle(reviews)

    split_index = int(percent * len(reviews))
    train = np.array(reviews[:split_index])
    test = np.array(reviews[split_index:])
    return train, test

def load_all():
    train_reviews = []
    test_reviews = []
    train_categories = []
    test_categories = []
    for category in CATEGORIES:
        train, test = load_reviews('db/reviews_' + category + '_5.json', 0.75)
        train_reviews.append(train)
        test_reviews.append(test)
        train_categories.append([category] * train.shape[0])
        test_categories.append([category] * test.shape[0])

    train_reviews = np.concatenate(train_reviews)
    test_reviews = np.concatenate(test_reviews)
    train_categories = np.concatenate(train_categories)
    test_categories = np.concatenate(test_categories)

    train_all = list(zip(train_reviews, train_categories))
    test_all = list(zip(test_reviews, test_categories))
    shuffle(train_all)
    shuffle(test_all)
    train_reviews, train_categories = zip(*train_all)
    test_reviews, test_categories = zip(*test_all)
    return train_reviews, test_reviews, np.array(train_categories), np.array(test_categories)

def parse_BOW(train_features, test_features):
    train_summary_corpus = []
    train_review_corpus = []
    train_final_features = []
    for i, feature in enumerate(train_features):
        train_summary_corpus.append(feature[5])
        train_review_corpus.append(feature[6])
        train_final_features.append(feature[0:5])

    test_summary_corpus = []
    test_review_corpus = []
    test_final_features = []
    for i, feature in enumerate(test_features):
        test_summary_corpus.append(feature[5])
        test_review_corpus.append(feature[6])
        test_final_features.append(feature[0:5])

    train_summary_BOW, test_summary_BOW = ps.sparsify(train_summary_corpus, test_summary_corpus)
    train_review_BOW, test_review_BOW = ps.sparsify(train_review_corpus, test_review_corpus)
    train_final_features = hstack([csr_matrix(train_final_features, dtype=np.int64), train_summary_BOW, train_review_BOW])
    test_final_features = hstack([csr_matrix(test_final_features, dtype=np.int64), test_summary_BOW, test_review_BOW])
    return train_final_features, test_final_features

if __name__ == '__main__':
    SKIP_LOAD_ALL = 1

    if SKIP_LOAD_ALL:
        train_x = load_npz('train_x.npz')
        test_x = load_npz('test_x.npz')
        train_y = np.load('train_y.npy')
        test_y = np.load('test_y.npy')
    else:
        train_x, test_x, train_y, test_y = load_all()
        train_x, test_x = parse_BOW(train_x, test_x)
        # print(train_x.shape)
        # print(test_x.shape)
        save_npz('train_x',train_x)
        save_npz('test_x',test_x)
        np.save('train_y',train_y)
        np.save('test_y',train_y)

    cf.train_predict_error(train_x, train_y, test_x, test_y, cf.train_naive_bayes)
    cf.train_predict_error(train_x, train_y, test_x, test_y, cf.train_logistic_regression)
    cf.train_predict_error(train_x, train_y, test_x, test_y, cf.train_svm)

    predictions = cf.predict_ensemble(train_x, train_y, test_x, [cf.train_naive_bayes, cf.train_logistic_regression, cf.train_svm])
    cf.error_ratio(predictions, test_y)
