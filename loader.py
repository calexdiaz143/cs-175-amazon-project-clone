from scipy.sparse import save_npz, load_npz
from random import shuffle
import numpy as np
import json
import parser

def get_helpful_percentage(ratio):
    if(ratio[1] == 0):
        return 50
    return int(100 * ratio[0] / ratio[1])

def get_category(path, percent):
    json_reviews = open(path).read().split('\n')[:-1]
    reviews = []
    for json_review in json_reviews:
        raw_review = json.loads(json_review)
        review = [
            1, 1, # TODO: figure out how to reduce the size of int(x, 36)
        	# int(raw_review['reviewerID'], 36),
        	# int(raw_review['asin'], 36),
        	get_helpful_percentage(raw_review['helpful']),
        	int(raw_review['overall']),
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

def get_categories(categories):
    train_reviews = []
    train_categories = []
    test_reviews = []
    test_categories = []
    for category in categories:
        train, test = get_category('db/reviews_' + category + '_5.json', 0.75)
        train_reviews.append(train)
        train_categories.append([category] * train.shape[0])
        test_reviews.append(test)
        test_categories.append([category] * test.shape[0])

    train_reviews = np.concatenate(train_reviews)
    train_categories = np.concatenate(train_categories)
    test_reviews = np.concatenate(test_reviews)
    test_categories = np.concatenate(test_categories)

    train_all = list(zip(train_reviews, train_categories))
    test_all = list(zip(test_reviews, test_categories))
    shuffle(train_all)
    shuffle(test_all)
    train_reviews, train_categories = zip(*train_all)
    test_reviews, test_categories = zip(*test_all)
    return train_reviews, np.array(train_categories), test_reviews, np.array(test_categories)

def save(train_X, train_Y, test_X, test_Y):
    save_npz('saved/train_X', train_X)
    np.save('saved/train_Y', train_Y)
    save_npz('saved/test_X', test_X)
    np.save('saved/test_Y', train_Y)

def load(categories, use_saved = False):
    if use_saved:
        train_X = load_npz('saved/train_X.npz')
        train_Y = np.load('saved/train_Y.npy')
        test_X = load_npz('saved/test_X.npz')
        test_Y = np.load('saved/test_Y.npy')
    else:
        train_X, train_Y, test_X, test_Y = get_categories(categories)
        train_X, test_X = parser.parse_BOW(train_X, test_X)
        save(train_X, train_Y, test_X, test_Y)
    return train_X, train_Y, test_X, test_Y
