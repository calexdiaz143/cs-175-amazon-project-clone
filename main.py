# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
from random import shuffle
import numpy as np
import json
import parsers as ps

SKIP_LOAD_ALL = True

CATEGORIES = [
	'Musical_Instruments',
	'Patio_Lawn_and_Garden'
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

def parse_BOW(features):
    summary_corpus = []
    review_corpus = []
    final_features = []
    for i, feature in enumerate(features):
        summary_corpus.append(feature[5])
        review_corpus.append(feature[6])
        final_features.append(feature[0:5])

    summary_BOW = ps.sparsify(summary_corpus)
    review_BOW = ps.sparsify(review_corpus)
    final_features = hstack([csr_matrix(final_features, dtype=np.int64), summary_BOW, review_BOW])
    return final_features

def train_naive_bayes(train_reviews):
    X, Y = zip(*train_reviews)
    classifier  = MultinomialNB().fit(X, Y)
    return classifier

if SKIP_LOAD_ALL:
    train_x = load_npz('train_x.npz')
    test_x = load_npz('test_x.npz')
    train_y = np.load('train_y.npy')
    test_y = np.load('test_y.npy')
else:
    train_x, test_x, train_y, test_y = load_all()
    train_x = parse_BOW(train_x)
    test_x = parse_BOW(test_x)
    save_npz('train_x',train_x)
    save_npz('test_x',test_x)
    np.save('train_y',train_y)
    np.save('test_y',train_y)
    
#classifier = train_naive_bayes(train)