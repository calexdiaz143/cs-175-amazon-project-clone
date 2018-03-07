import memo, trainer, tester

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

if __name__ == '__main__':
    # load data and train classifiers
    LOAD_SAVED = True
    OVERWRITE_SAVED = True
    train_X, train_Y, test_X, test_Y, summary_cv, review_cv = memo.get_data(LOAD_SAVED, OVERWRITE_SAVED, CATEGORIES, 0.75, 1000)
    clf_nb = memo.get_classifier(LOAD_SAVED, OVERWRITE_SAVED, trainer.naive_bayes, train_X, train_Y, 'clf_nb')
    clf_lr = memo.get_classifier(LOAD_SAVED, OVERWRITE_SAVED, trainer.logistic_regression, train_X, train_Y, 'clf_lr')
    clf_svm = memo.get_classifier(LOAD_SAVED, OVERWRITE_SAVED, trainer.svm, train_X, train_Y, 'clf_svm')

    # get error rate
    prd_nb = clf_nb.predict(test_X)
    prd_lr = clf_lr.predict(test_X)
    prd_svm = clf_svm.predict(test_X)

    err_nb = tester.error_ratio(test_Y, prd_nb)
    err_lr = tester.error_ratio(test_Y, prd_lr)
    err_svm = tester.error_ratio(test_Y, prd_svm)

    print(err_nb)
    print(err_lr)
    print(err_svm)

    predictions = [prd_nb, prd_lr, prd_svm]
    prd_ensemble = tester.predict_ensemble(test_X, predictions)
    err_ensemble = tester.error_ratio(test_Y, prd_ensemble)

    print(err_ensemble)

    # check for overfitting
    prd_nb = clf_nb.predict(train_X)
    prd_lr = clf_lr.predict(train_X)
    prd_svm = clf_svm.predict(train_X)

    err_nb = tester.error_ratio(train_Y, prd_nb)
    err_lr = tester.error_ratio(train_Y, prd_lr)
    err_svm = tester.error_ratio(train_Y, prd_svm)

    print(err_nb)
    print(err_lr)
    print(err_svm)

    predictions = [prd_nb, prd_lr, prd_svm]
    prd_ensemble = tester.predict_ensemble(train_X, predictions)
    err_ensemble = tester.error_ratio(test_Y, prd_ensemble)

    print(err_ensemble)
