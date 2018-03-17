from __future__ import division
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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

# Normalized Confusion Matrix: Heavily based on the scikit-learn example here:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def disp_conf_matrix(Y, Y_hat, classes):
    cm = confusion_matrix(Y, Y_hat,labels=classes)
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.show()