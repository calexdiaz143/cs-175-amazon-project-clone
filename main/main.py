import loader
import trainer
import tester

CATEGORIES = [
    # "Books",                          # 8,898,041
    # "Electronics",                    # 1,689,188
    # "Movies_and_TV",                  # 1,697,533
    # "CDs_and_Vinyl",                  # 1,097,592
    # "Clothing_Shoes_and_Jewelry",     # 278,677
    # "Home_and_Kitchen",               # 551,682
    # "Kindle_Store",                   # 82,619
    # "Sports_and_Outdoors",            # 296,337
    # "Cell_Phones_and_Accessories",    # 194,439
    # "Health_and_Personal_Care",       # 346,355
    # "Toys_and_Games",                 # 167,597
    # "Video_Games",                    # 231,780
    # "Tools_and_Home_Improvement",     # 134,476
    # "Beauty",                         # 198,502
    # "Apps_for_Android",               # 752,937
    # "Office_Products",                # 53,258
    # "Pet_Supplies",                   # 157,836
    "Automotive",                     # 20,473
    "Grocery_and_Gourmet_Food",       # 151,254
    "Patio_Lawn_and_Garden",          # 13,272
    "Baby",                           # 160,792
    "Digital_Music",                  # 64,706
    "Musical_Instruments",            # 10,261
    "Amazon_Instant_Video"            # 37,126
]

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = loader.load(CATEGORIES, percent=0.75, cutoff=100, use_saved=False, overwrite_saved=True)

    # Option 1: train and save the classifers
    classifier_naive_bayes = trainer.naive_bayes(train_X, train_Y)
    classifier_logistic_regression = trainer.logistic_regression(train_X, train_Y)
    classifier_svm = trainer.svm(train_X, train_Y)

    trainer.save(classifier_naive_bayes, "static/clf_nb")
    trainer.save(classifier_logistic_regression, "static/clf_log")
    trainer.save(classifier_svm, "static/clf_svm")

    # PREDICT A SINGLE CATEGORY
    def predict(review, classifier):


    predict()
    # PREDICT A SINGLE CATEGORY

    # Option 2: just load the saved classifiers
    # classifier_naive_bayes = trainer.load('clf_nb')
    # classifier_logistic_regression = trainer.load('clf_log')
    # classifier_svm = trainer.load('clf_svm')

    tester.predict_error(test_X, test_Y, classifier_naive_bayes)
    tester.predict_error(test_X, test_Y, classifier_logistic_regression)
    tester.predict_error(test_X, test_Y, classifier_svm)

    # TODO: save the predictions above to use in the ensemble below, maybe, idk

    classifiers = [
        classifier_naive_bayes,
        classifier_logistic_regression,
        classifier_svm
    ]
    predictions = tester.predict_ensemble(test_X, classifiers)
    tester.error_ratio(predictions, test_Y)
