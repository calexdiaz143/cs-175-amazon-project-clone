import loader
import trainer
import tester

CATEGORIES = [
    "Books",
    "Electronics",
    "Movies_and_TV",
    "CDs_and_Vinyl",
    "Clothing,_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Sports_and_Outdoors",
    "Cell_Phones_and_Accessories",
    "Health_and_Personal_Care",
    "Toys_and_Games",
    "Video_Games",
    "Tools_and_Home_Improvement",
    "Beauty",
    "Apps_for_Android",
    "Office_Products",
    "Pet_Supplies",
    "Automotive",
    "Grocery_and_Gourmet_Food",
    "Patio,_Lawn_and_Garden",
    "Baby",
    "Digital_Music",
    "Musical_Instruments",
    "Amazon_Instant_Video"
]

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = loader.load(CATEGORIES, True)

    tester.train_predict_error(train_X, train_Y, test_X, test_Y, trainer.naive_bayes)
    tester.train_predict_error(train_X, train_Y, test_X, test_Y, trainer.logistic_regression)
    tester.train_predict_error(train_X, train_Y, test_X, test_Y, trainer.svm)

    # TODO: save the classifiers as files (hopefully this is possibly; it's the only way the website will work)
    # clf = trainer.naive_bayes(train_X, train_Y)
    # import pickle
    # pickle.dump(clf, open('saved/clf.pkl', 'wb'))
    # clf = pickle.load(open('saved/clf.pkl', 'rb'))

    # TODO: save the predictions above to use in the ensemble below, maybe, idk

    classifiers = [
        trainer.naive_bayes,
        trainer.logistic_regression,
        trainer.svm
    ]
    predictions = tester.predict_ensemble(train_X, train_Y, test_X, classifiers)
    tester.error_ratio(predictions, test_Y)
