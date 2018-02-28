from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def idToNum(idString):
    return 1 #int(idString,36)

def helpfulToNum(helpfulTuple):
    if(helpfulTuple[1] == 0):
        helpfulness = 50
    else:
        helpfulness = int(100*(helpfulTuple[0]/helpfulTuple[1]))
    return helpfulness

def overallToNum(overallRating):
    return int(overallRating)

def sparsify(train_corpus, test_corpus):
    countVectorizer = CountVectorizer(stop_words=stopwords.words('english'), min_df = 5)
    train_text_counts = countVectorizer.fit_transform(train_corpus)
    test_text_counts = countVectorizer.transform(test_corpus)
    # add tf-idf feature later on
    return train_text_counts, test_text_counts
