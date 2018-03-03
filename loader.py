from scipy.sparse import save_npz, load_npz
from random import shuffle
import numpy as np
import json
import parser

def get_category(path, percent, cutoff=-1):
    content = open(path)
    reviews = []
    for json_review in content.readlines():
        raw_review = json.loads(json_review)
        review = parser.parse_review(raw_review)
        if review[2] >= 0.5:
            reviews.append(review)
        
        if cutoff == 0:
            break
        cutoff -= 1

    shuffle(reviews)

    split_index = int(percent * len(reviews))
    train = np.array(reviews[:split_index])
    test = np.array(reviews[split_index:])
    return train, test

def get_categories(categories, percent, cutoff=-1):
    train_reviews = []
    train_categories = []
    test_reviews = []
    test_categories = []
    for category in categories:
        train, test = get_category('db/reviews_' + category + '_5.json', percent, cutoff)
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

def load(categories, percent=0.75, cutoff=-1, use_saved=False, overwrite_saved=True):
    if use_saved:
        train_X = load_npz('saved/train_X.npz')
        train_Y = np.load('saved/train_Y.npy')
        test_X = load_npz('saved/test_X.npz')
        test_Y = np.load('saved/test_Y.npy')
    else:
        train_X, train_Y, test_X, test_Y = get_categories(categories, percent, cutoff)
        train_X, test_X = parser.parse_BOW(train_X, test_X)
        if overwrite_saved:
            save(train_X, train_Y, test_X, test_Y)
    return train_X, train_Y, test_X, test_Y
