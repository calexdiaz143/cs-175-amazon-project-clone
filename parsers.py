from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def idToNum(idString):
    return int(idString,36)

def helpfulToNum(helpfulTuple):
    if(helpfulTuple[1] == 0):
        helpfulness = 0.5
    else:
        helpfulness = helpfulTuple[0]/helpfulTuple[1]
    return helpfulness

def overallToNum(overallRating):
    return int(overallRating)

def sparsify(corpus):
    countVectorizer = CountVectorizer(stop_words=stopwords.words('english'), min_df = 2) 
    textCounts = countVectorizer.fit_transform(corpus)  
    # add tf-idf feature later on
    return textCounts