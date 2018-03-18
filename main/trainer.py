from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

def naive_bayes(X, Y):
    '''Trains and returns a Multinomial Naive Bayes classifier with the given data and labels.'''
    # Default Parameters
    #   alpha = 1
    classifier = MultinomialNB(alpha=0.0001).fit(X, Y)
    return classifier

def bernoulli_naive_bayes(X, Y):
    '''Trains and returns a Bernoulli Naive Bayes classifier with the given data and labels.'''
    # Default Parameters
    #   alpha = 1
    classifier = BernoulliNB(alpha=0.0001).fit(X, Y)
    return classifier

def logistic_regression(X, Y):
    '''Trains and returns a Logistic Regression classifier with the given data and labels.'''
    # Default Parameters
    #   penalty = 'l2'
    #   C = 1
    classifier = LogisticRegression(penalty='l1', C=1000).fit(X, Y)
    return classifier

def svm(X, Y):
    '''Trains and returns a Support Vector Machine classifier with the given data and labels.'''
    # Default Parameters
    #   x
    classifier = SVC().fit(X, Y)
    return classifier

def knn(X, Y):
    '''Trains and returns a K-Nearest Neighbors classifier with the given data and labels.'''
    # Default Parameters
    #   x
    classifier = KNeighborsClassifier(n_neighbors=1000, weights='distance').fit(X, Y)
    return classifier

def mlp(X, Y):
    '''Trains and returns a Multilayer Perceptron classifier with the given data and labels.'''
    # Default Parameters
    #   hidden_layer_sizes = (100,)
    #   alpha = 0.0001
    #   learning_rate = 'constant'
    classifier = MLPClassifier(hidden_layer_sizes=(3,3), alpha=1).fit(X, Y)
    return classifier

def random_forest(X, Y):
    '''Trains and returns a Random Forest classifier with the given data and labels.'''
    # Default Parameters
    #   n_estimators = 10
    #   max_features = 'auto'
    #   max_depth = None
    #   max_leaf_nodes = None
    #   min_samples_leaf = 1
    classifier = RandomForestClassifier(n_estimators=200).fit(X, Y)
    return classifier

def adaboost(X, Y):
    '''Trains and returns an AdaBoost classifier with the given data and labels.'''
    # Default Parameters
    #   n_estimators = 50
    #   learning_rate = 1
    classifier = AdaBoostClassifier(n_estimators=200, learning_rate=0.5).fit(X, Y)
    return classifier

def gradient_boost(X, Y):
    '''Trains and returns a Gradient Boosting classifier with the given data and labels.'''
    # Default Parameters
    #   n_estimators = 100
    #   max_depth = 3
    #   learning_rate = 0.1
    classifier = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1).fit(X, Y)
    return classifier
