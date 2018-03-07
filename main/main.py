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
    # for all memo.get() functions, the first two parameters are (LOAD_SAVED, OVERWRITE_SAVED)
    # hopefully those names are self-explanatory
    train_X, train_Y, test_X, test_Y, summary_CV, review_CV = memo.get_data(True, False, CATEGORIES, 0.75, 100)
    clf_NB = memo.get_classifier(True, False, trainer.naive_bayes, train_X, train_Y, 'clf_NB')
    clf_LR = memo.get_classifier(True, False, trainer.logistic_regression, train_X, train_Y, 'clf_LR')
    clf_SVM = memo.get_classifier(True, False, trainer.svm, train_X, train_Y, 'clf_SVM')

    prd_NB = clf_NB.predict(test_X)
    prd_LR = clf_LR.predict(test_X)
    prd_SVM = clf_SVM.predict(test_X)

    err_NB = tester.error_ratio(test_Y, prd_NB)
    err_LR = tester.error_ratio(test_Y, prd_LR)
    err_SVM = tester.error_ratio(test_Y, prd_SVM)

    print(err_NB)
    print(err_LR)
    print(err_SVM)

    predictions = [prd_NB, prd_LR, prd_SVM]
    prd_E = tester.predict_ensemble(test_X, predictions)
    err_E = tester.error_ratio(test_Y, prd_E)

    print(err_E)
