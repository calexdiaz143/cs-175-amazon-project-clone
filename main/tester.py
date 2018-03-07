from __future__ import division
from collections import Counter

def get_majority(prediction_list, index):
    prediction = [pred[index] for pred in prediction_list]
    majority = Counter(prediction).most_common()
    return majority[0][0]

def predict_ensemble(X, prediction_list):
    ensemble_predictions = []
    for i in range(X.shape[0]):
        ensemble_predictions.append(get_majority(prediction_list, i))
    return ensemble_predictions

def error_ratio(Y, predictions):
    error = 0
    for y, prediction in zip(Y, predictions):
        if y != prediction:
            error += 1
    return error / Y.shape[0]
