import loader
import trainer
import tester

CATEGORIES = [
    'Musical_Instruments',
    'Patio_Lawn_and_Garden',
    'Amazon_Instant_Video'
]

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = loader.load(CATEGORIES)

    tester.train_predict_error(train_X, train_Y, test_X, test_Y, trainer.naive_bayes)
    tester.train_predict_error(train_X, train_Y, test_X, test_Y, trainer.logistic_regression)
    tester.train_predict_error(train_X, train_Y, test_X, test_Y, trainer.svm)

    # TODO: save the predictions above to use in the ensemble below, maybe, idk
    # TODO: save the classifiers as files (hopefully this is possibly; it's the only way the website will work)

    classifiers = [
        trainer.naive_bayes,
        trainer.logistic_regression,
        trainer.svm
    ]
    predictions = tester.predict_ensemble(train_X, train_Y, test_X, classifiers)
    tester.error_ratio(predictions, test_Y)
