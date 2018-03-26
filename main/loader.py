from random import shuffle
import numpy as np
import json

SHOW_MESSAGES = False

def parse_review(raw_review):
    '''
    Transforms original review data from a dictionary to a list.
    Returns the data list in the order [summary, review, helpfulness, rating, time].

    raw_review: the review data in the original dictionary format
    '''
    return [
        raw_review['summary'],
        raw_review['reviewText'],
        2 * int(raw_review['helpful'][0]) - int(raw_review['helpful'][1]),
        int(raw_review['overall']),
        raw_review['unixReviewTime']
    ]

def load_category(path, percent, cutoff):
    '''
    Loads and processes original review data line by line from the specified category file.
    Returns a tuple (train, test) with train data and test data.

    path:    the filepath of the category file
    percent: the percent of data to use as train data (the remainder is test data)
    cutoff:  the number of reviews to use from the category file (skips unhelpful reviews)
    '''
    reviews = []
    with open(path, 'r') as content:
        for json_review in content:
            raw_review = json.loads(json_review)
            review = parse_review(raw_review)
            if review[2] >= 0:
                reviews.append(review)

                cutoff -= 1
                if cutoff == 0:
                    break
    shuffle(reviews)

    split_index = int(percent * len(reviews))
    train = np.array(reviews[:split_index])
    test = np.array(reviews[split_index:])
    return train, test

def load_categories(categories, percent, cutoff):
    '''
    Loads and processes original review data from the specified category files.
    Returns a tuple (train_X, train_Y, test_X, test_Y) where X is the review data and Y is the label.

    categories: a list of category file basenames
    percent:    the percent of data to use as train data (the remainder is test data)
    cutoff:  the number of reviews to use from the category file (skips unhelpful reviews)
    '''
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
        if SHOW_MESSAGES:
            import message
            message.say('The {} category is ready.'.format(category))

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
