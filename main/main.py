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
    OVERWRITE_SAVED = False
    train_X, train_Y, test_X, test_Y, summary_cv, review_cv = memo.get_data(LOAD_SAVED, OVERWRITE_SAVED, CATEGORIES, 0.75, 100)
    # clf_svm = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.svm, train_X, train_Y, 'clf_svm')
    # clf_knn = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.knn, train_X, train_Y, 'clf_knn')
    # clf_mlp = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.mlp, train_X, train_Y, 'clf_mlp')
    # clf_ada = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.adaboost, train_X, train_Y, 'clf_ada')
    clf_nb = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.naive_bayes, train_X, train_Y, 'clf_nb')
    clf_bnb = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.bernoulli_naive_bayes, train_X, train_Y, 'clf_bnb')
    clf_lr = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.logistic_regression, train_X, train_Y, 'clf_lr')
    clf_rf = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.random_forest, train_X, train_Y, 'clf_rf')
    clf_gb = memo.get_classifier(not LOAD_SAVED, not OVERWRITE_SAVED, trainer.gradient_boost, train_X, train_Y, 'clf_gb')

    # get error rate

    # test_prd_svm = clf_svm.predict(test_X)
    # test_prd_knn = clf_knn.predict(test_X)
    # test_prd_mlp = clf_mlp.predict(test_X)
    # test_prd_ada = clf_ada.predict(test_X)
    test_prd_nb = clf_nb.predict(test_X)
    test_prd_bnb = clf_bnb.predict(test_X)
    test_prd_lr = clf_lr.predict(test_X)
    test_prd_rf = clf_rf.predict(test_X)
    test_prd_gb = clf_gb.predict(test_X)

    # test_err_svm = tester.error_ratio(test_Y, test_prd_svm)
    # test_err_knn = tester.error_ratio(test_Y, test_prd_knn)
    # test_err_mlp = tester.error_ratio(test_Y, test_prd_mlp)
    # test_err_ada = tester.error_ratio(test_Y, test_prd_ada)
    test_err_nb = tester.error_ratio(test_Y, test_prd_nb)
    test_err_bnb = tester.error_ratio(test_Y, test_prd_bnb)
    test_err_lr = tester.error_ratio(test_Y, test_prd_lr)
    test_err_rf = tester.error_ratio(test_Y, test_prd_rf)
    test_err_gb = tester.error_ratio(test_Y, test_prd_gb)

    test_predictions = [test_prd_nb, test_prd_bnb, test_prd_lr, test_prd_rf, test_prd_gb]
    test_prd_ensemble = tester.predict_ensemble(test_X, test_predictions)
    test_err_ensemble = tester.error_ratio(test_Y, test_prd_ensemble)

    # check for overfitting

    # train_prd_svm = clf_svm.predict(train_X)
    # train_prd_knn = clf_knn.predict(train_X)
    # train_prd_mlp = clf_mlp.predict(train_X)
    # train_prd_ada = clf_ada.predict(train_X)
    train_prd_nb = clf_nb.predict(train_X)
    train_prd_bnb = clf_bnb.predict(train_X)
    train_prd_lr = clf_lr.predict(train_X)
    train_prd_rf = clf_rf.predict(train_X)
    train_prd_gb = clf_gb.predict(train_X)

    # train_err_svm = tester.error_ratio(train_Y, train_prd_svm)
    # train_err_knn = tester.error_ratio(train_Y, train_prd_knn)
    # train_err_mlp = tester.error_ratio(train_Y, train_prd_mlp)
    # train_err_ada = tester.error_ratio(train_Y, train_prd_ada)
    train_err_nb = tester.error_ratio(train_Y, train_prd_nb)
    train_err_bnb = tester.error_ratio(train_Y, train_prd_bnb)
    train_err_lr = tester.error_ratio(train_Y, train_prd_lr)
    train_err_rf = tester.error_ratio(train_Y, train_prd_rf)
    train_err_gb = tester.error_ratio(train_Y, train_prd_gb)

    train_predictions = [train_prd_nb, train_prd_bnb, train_prd_lr, train_prd_rf, train_prd_gb]
    train_prd_ensemble = tester.predict_ensemble(train_X, train_predictions)
    train_err_ensemble = tester.error_ratio(train_Y, train_prd_ensemble)

    # memo.save_txt('static/lr_error_', '''
    # Test Error LR : {:10.6f}
    # Train Error LR : {:10.6f}
    # '''.format(test_err_lr, train_err_lr))

    # print('{:10.6f}\n{:10.6f}'.format(test_err_svm, train_err_svm))
    # print('{:10.6f}\n{:10.6f}'.format(test_err_knn, train_err_knn))
    # print('{:10.6f}\n{:10.6f}'.format(test_err_mlp, train_err_mlp))
    # print('{:10.6f}\n{:10.6f}'.format(test_err_ada, train_err_ada))
    print('{:10.6f}\n{:10.6f}'.format(test_err_nb, train_err_nb))
    print('{:10.6f}\n{:10.6f}'.format(test_err_bnb, train_err_bnb))
    print('{:10.6f}\n{:10.6f}'.format(test_err_lr, train_err_lr))
    print('{:10.6f}\n{:10.6f}'.format(test_err_rf, train_err_rf))
    print('{:10.6f}\n{:10.6f}'.format(test_err_gb, train_err_gb))
    print('{:10.6f}\n{:10.6f}'.format(test_err_ensemble, train_err_ensemble))

    # memo.save_txt('static/error', '''
    # Test Error NB : {:10.6f}
    # Test Error LR : {:10.6f}
    # Test Error SVM: {:10.6f}
    # Test Error EN : {:10.6f}
    #
    # Train Error NB : {:10.6f}
    # Train Error LR : {:10.6f}
    # Train Error SVM: {:10.6f}
    # Train Error EN : {:10.6f}
    # '''.format(
    #     test_err_nb, test_err_lr, test_err_svm, test_err_ensemble,
    #     train_err_nb, train_err_lr, train_err_svm, train_err_ensemble
    # ))
