from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import Counter

def train_naive_bayes(X, Y):
    classifier  = MultinomialNB().fit(X, Y)
    return classifier

def train_logistic_regression(X, Y):
    classifier  = LogisticRegression().fit(X, Y)
    return classifier

def train_svm(X, Y):
    classifier  = SVC().fit(X, Y)
    return classifier

def get_majority(prediction_list, index):
    prediction = [pred[index] for pred in prediction_list]
    majority = Counter(prediction).most_common()
    return majority[0][0]

def predict_ensemble(train_X, train_Y, test_X, classifiers):
    predictions = []
    ensemble_predictions = []
    for clf in classifiers:
        classifier = clf(train_X, train_Y)
        predictions.append(classifier.predict(test_X))

    for i in range(test_X.shape[0]):
        ensemble_predictions.append(get_majority(predictions,i))
    return ensemble_predictions

def error_ratio(Y_hat, Y):
    error = 0
    for y_hat, y in zip(Y_hat, Y):
        if y_hat != y:
            error += 1
    return error / Y.shape[0]

def train_predict_error(train_X, train_Y, test_X, test_Y, trainer):
    classifier = trainer(train_X, train_Y)
    prediction = classifier.predict(test_X)
    error = error_ratio(prediction, test_Y)
    print(error)
