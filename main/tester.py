from __future__ import division
from collections import Counter

def get_majority(prediction_list, index):
    prediction = [pred[index] for pred in prediction_list]
    majority = Counter(prediction).most_common()
    return majority[0][0]

def predict_ensemble(test_X, classifiers):
    predictions = []
    ensemble_predictions = []
    for classifier in classifiers:
        prediction = classifier.predict(test_X)
        predictions.append(prediction)

    for i in range(test_X.shape[0]):
        ensemble_predictions.append(get_majority(predictions, i))
    return ensemble_predictions

def error_ratio(Y_hat, Y):
    error = 0
    for y_hat, y in zip(Y_hat, Y):
        if y_hat != y:
            error += 1
    return error / Y.shape[0]

def predict_error(test_X, test_Y, classifer):
    prediction = classifer.predict(test_X)
    error = error_ratio(prediction, test_Y)
    print(error)
