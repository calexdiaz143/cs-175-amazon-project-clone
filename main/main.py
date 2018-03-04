CATEGORIES = [
    "Books",                          # 8,898,041
    "Electronics",                    # 1,689,188
    "Movies_and_TV",                  # 1,697,533
    "CDs_and_Vinyl",                  # 1,097,592
    "Clothing_Shoes_and_Jewelry",     # 278,677
    "Home_and_Kitchen",               # 551,682
    "Kindle_Store",                   # 82,619
    "Sports_and_Outdoors",            # 296,337
    "Cell_Phones_and_Accessories",    # 194,439
    "Health_and_Personal_Care",       # 346,355
    "Toys_and_Games",                 # 167,597
    "Video_Games",                    # 231,780
    "Tools_and_Home_Improvement",     # 134,476
    "Beauty",                         # 198,502
    "Apps_for_Android",               # 752,937
    "Office_Products",                # 53,258
    "Pet_Supplies",                   # 157,836
    "Automotive",                     # 20,473
    "Grocery_and_Gourmet_Food",       # 151,254
    "Patio_Lawn_and_Garden",          # 13,272
    "Baby",                           # 160,792
    "Digital_Music",                  # 64,706
    "Musical_Instruments",            # 10,261
    "Amazon_Instant_Video"            # 37,126
]

def predict(review, classifier, summary_cv_path, review_cv_path):
    import parser, trainer, tester
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle

    review = parser.parse_review(review)

    summary_corpus = [review[5]]
    review_corpus = [review[6]]
    final_features = [review[0:5]]

    summary_cv = pickle.load(open(summary_cv_path, 'rb'))
    summary_BOW = summary_cv.transform(summary_corpus)

    review_cv = pickle.load(open(review_cv_path, 'rb'))
    review_BOW = review_cv.transform(review_corpus)

    final_features = hstack([csr_matrix(final_features, dtype=np.int64), summary_BOW, review_BOW])

    prediction = classifier.predict(final_features)
    print(type(test_X))
    print(type(final_features))
    print(prediction)

def predict_django(review, classifier, summary_cv_path, review_cv_path): # exact same as above, but for the website
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    import pickle

    summary_corpus = [review[5]]
    review_corpus = [review[6]]
    final_features = [review[0:5]]

    summary_cv = pickle.load(open(summary_cv_path, 'rb'))
    summary_BOW = summary_cv.transform(summary_corpus)

    review_cv = pickle.load(open(review_cv_path, 'rb'))
    review_BOW = review_cv.transform(review_corpus)

    final_features = hstack([csr_matrix(final_features, dtype=np.int64), summary_BOW, review_BOW])

    prediction = classifier.predict(final_features)

if __name__ == '__main__':
    import loader
    import trainer
    import tester

    train_X, train_Y, test_X, test_Y = loader.load(CATEGORIES, percent=0.75, cutoff=100, use_saved=False, overwrite_saved=False)

    # Option 1: train and save the classifers
    classifier_naive_bayes = trainer.naive_bayes(train_X, train_Y)
    classifier_logistic_regression = trainer.logistic_regression(train_X, train_Y)
    classifier_svm = trainer.svm(train_X, train_Y)

    trainer.save(classifier_naive_bayes, "static/clf_nb")
    trainer.save(classifier_logistic_regression, "static/clf_log")
    trainer.save(classifier_svm, "static/clf_svm")

    # PREDICT A SINGLE CATEGORY

    predict({
    	"reviewerID": "A2NYK9KWFMJV4Y",
    	"asin": "B0002E5518",
    	"reviewerName": "Mike Tarrani \"Jazz Drummer\"",
    	"helpful": [1, 1],
    	"reviewText": "One thing I love about this extension bar is it will fit over the relatively large diameter down tubes on my hi-hat stands. I use anOn Stage Microphone 13-inch Gooseneckto connect to the bar, then I attach either aNady DM70 Drum and Instrument Microphoneor DM80 microphone. The bar-gooseneck arrangement is sturdy enough for that mic model.This also works well for mounting on microphone stands. I use it and a shorter gooseneck to mount the DM70 for tenor and alto saxophones, or a DM80 for baritones. Again, it works perfectly for my situations.I always keep a few of these, plus various size goosenecks, just in case I need to mount an additional microphone and I am short on stands. It's one more tool to make set up less stressful.Trust me, when you need one (and chances are you will if you're a drummer or a sound tech) you will thank yourself for having the foresight to purchase it for situations I cited above and those that I have not foreseen.",
    	"overall": 5.0,
    	"summary": "Highly useful - especially for drummers (and saxophonists)",
    	"unixReviewTime": 1370822400,
    	"reviewTime": "06 10, 2013"
    }, classifier_logistic_regression, 'static/summary_cv.pkl', 'static/review_cv.pkl')

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
