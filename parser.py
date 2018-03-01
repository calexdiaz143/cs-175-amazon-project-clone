# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from nltk.corpus import stopwords
import numpy as np

def sparsify(train_corpus, test_corpus):
    countVectorizer = CountVectorizer(stop_words=stopwords.words('english'), min_df=5)
    train_text_counts = countVectorizer.fit_transform(train_corpus)
    test_text_counts = countVectorizer.transform(test_corpus)
    # add tf-idf feature later on
    return train_text_counts, test_text_counts

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

    train_summary_BOW, test_summary_BOW = sparsify(train_summary_corpus, test_summary_corpus)
    train_review_BOW, test_review_BOW = sparsify(train_review_corpus, test_review_corpus)
    train_final_features = hstack([csr_matrix(train_final_features, dtype=np.int64), train_summary_BOW, train_review_BOW])
    test_final_features = hstack([csr_matrix(test_final_features, dtype=np.int64), test_summary_BOW, test_review_BOW])
    return train_final_features, test_final_features