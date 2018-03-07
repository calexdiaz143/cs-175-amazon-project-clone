from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def naive_bayes(X, Y):
    classifier = MultinomialNB().fit(X, Y)
    return classifier

def logistic_regression(X, Y):
    classifier = LogisticRegression(penalty='l1', C=1000).fit(X, Y)
    return classifier

def svm(X, Y):
    classifier = SVC().fit(X, Y)
    return classifier
