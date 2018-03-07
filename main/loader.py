from random import shuffle
import numpy as np
import json

def get_helpful_percentage(ratio): #TODO: change this, and maybe put this back in parse_review
    if(ratio[1] == 0):
        return 50
    return int(100 * ratio[0] / ratio[1])

def parse_review(raw_review): # TODO: maybe put this back in get_category
    return [
        raw_review['summary'],
        raw_review['reviewText'],
        1, 1, # TODO: figure out how to reduce the size of int(x, 36)
        # int(raw_review['reviewerID'], 36),
        # int(raw_review['asin'], 36),
        get_helpful_percentage(raw_review['helpful']),
        int(raw_review['overall']),
        raw_review['unixReviewTime']
    ]

def load_category(path, percent, cutoff):
    content = open(path)
    reviews = []
    for json_review in content.readlines():
        raw_review = json.loads(json_review)
        review = parse_review(raw_review)
        if review[4] >= 0.5:
            reviews.append(review)

        if cutoff == 0:
            break
        cutoff -= 1

    shuffle(reviews)

    split_index = int(percent * len(reviews))
    train = np.array(reviews[:split_index])
    test = np.array(reviews[split_index:])
    return train, test

def load_categories(categories, percent, cutoff):
    train_reviews = []
    train_categories = []
    test_reviews = []
    test_categories = []
    for category in categories:
        train, test = load_category('db/reviews_' + category + '_5.json', percent, cutoff)
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
