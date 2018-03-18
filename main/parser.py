# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from nltk.corpus import stopwords
import numpy as np

def sparsify(train_corpus, test_corpus):
    '''Transforms the corpora from the train and test data.'''
    # TODO: ngrams
    countVectorizer = CountVectorizer(stop_words=stopwords.words('english'), min_df=5)
    train_text_counts = countVectorizer.fit_transform(train_corpus)
    test_text_counts = countVectorizer.transform(test_corpus)
    # add tf-idf features later on
    return train_text_counts, test_text_counts, countVectorizer

def separate_features(X):
    '''Separates textual features from other features in the parsed review data.'''
    summary = []
    review = []
    other = []
    for features in X:
        summary.append(features[0])
        review.append(features[1])
        other.append(features[2:])
    return summary, review, other

def fit_transform(train_X, test_X):
    '''Fits and transforms textual features into BOWs so classifiers can handle them.'''
    train_X_summary, train_X_review, train_X_other = separate_features(train_X)
    test_X_summary, test_X_review, test_X_other = separate_features(test_X)

    train_X_summary_BOW, test_X_summary_BOW, summary_vectorizer = sparsify(train_X_summary, test_X_summary)
    train_X_review_BOW, test_X_review_BOW, review_vectorizer = sparsify(train_X_review, test_X_review)

    train_X_final = hstack([train_X_summary_BOW, train_X_review_BOW, csr_matrix(train_X_other, dtype=np.int64)])
    test_X_final = hstack([test_X_summary_BOW, test_X_review_BOW, csr_matrix(test_X_other, dtype=np.int64)])
    return train_X_final, test_X_final, summary_vectorizer, review_vectorizer

def transform(X, summary_vectorizer, review_vectorizer):
    '''Transforms textual features into BOWs so classifiers can handle them.'''
    X_summary, X_review, X_other = separate_features(X)
    X_summary_BOW = summary_vectorizer.transform(X_summary)
    X_review_BOW = review_vectorizer.transform(X_review)
    return hstack([X_summary_BOW, X_review_BOW, csr_matrix(X_other, dtype=np.int64)])
