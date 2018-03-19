Project 18 (Amazon Category Predictor) File Descriptions

README.txt - contains a one-line description of each file in this project/ directory
project.ipynb - a Jupyter notebook using Python 3 to run a demonstration of our project
src/ - all of the code developed for our project
    main.py - the main file of the project which loads, saves, trains, and tests Amazon review data; if imported, this file includes only a list of review categories
    memo.py - includes functions which simplify the process loading and overwriting saved data (from static/)
    loader.py - includes functions for loading the original Amazon review data (from db/)
    parser.py - includes functions for transforming reviews' textual data into bag-of-words representation (via count vectorizer) so classifiers can process them
    trainer.py - includes functions for constructing classifiers
    tester.py - includes functions for returning classifier predictions and evaluation data (error rates, etc)
    __init__.py - a file that allows this directory to be used as a Python module (for the website demo)
    db/ - a directory containing the original Amazon review data from Julian McAuley of UCSD (contents not submitted due to large file size)
    static/ - a directory containing saved classifiers, vectorizers, and data
        clf_*.pkl - a saved classifier (some not submitted due to large file size)
        review_cv.pkl - the saved count vectorizer for review text
        summary_cv.pkl - the saved count vectorizer for review summaries
        train_X.pkl - train data (review features)
        train_Y.pkl - train data (review categories)
        test_X.pkl - test data (review features)
        test_Y.pkl - test data (review categories)
