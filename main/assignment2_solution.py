# script for Assignment 2 - Part 2, CS 175, Winter 2016, modified from
# the scikit-learn "working with text data" tutorial, online at
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#working-with-text-data

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import FreqDist
from nltk.corpus import stopwords
import numpy as np

#############
# Problem 1 #
#############
def load_news_dataset(n_categories):
    """
    Loads both training and test data from the 20 newsgroups dataset..

    Training Data:
        After loading the training news data with 20 groups,
        get the 'n_categories' most frequent categories in 'twenty_train_all' using 'FreqDist()'.
        Load the list of news data (name it 'twenty_train') again, but only including the ones that
        match those categories.

    Test Data:
        Load the test news data that have same categories as training data and name it as 'twenty_test'.

    Parameters
    ----------
    n_categories : int
        Only data from the n_categories most frequent categories will be loaded from the dataset.

    Returns
    -------
    Tuple(sklearn.datasets.base.Bunch)
        Returns both training and test data as sklearn.datasets.base.Bunch objects.
        i.e.:(twenty_train, twenty_test).

    Examples
    --------
    >>> twenty_train, twenty_test = load_news_dataset(n_categories)
    >>> type(twenty_train)
    <class 'sklearn.datasets.base.Bunch'>
    """



    # Download the 20 newsgroups for training set
    # (this can take a while the first time it is executed....)
    #print('\nLoading the full 20 newsgroups data set.....\n')
    print('Loading the newsgroups data set...\n')
    twenty_train_all = fetch_20newsgroups(subset='train')

    ### YOUR SOLUTION STARTS HERE### 

    categories = [twenty_train_all.target_names[t]
                  for t, c in FreqDist(twenty_train_all.target).most_common(n_categories)] 

    twenty_train =   fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=22) 
    twenty_test =    fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=22)

    return twenty_train, twenty_test
    ### END SOLUTIONS ###pass


#############
# Problem 2 #
#############
def extract_text_features(train_data, test_data, min_docs ):
    """
    Returns two types of training and test data features.
        1) Bags of words (BOWs): X_train_counts, X_test_counts
        2) Term Frequency times Inverse Document Frequency (tf-idf): X_train_tfidf, X_test_tfidf

    How to create BOW features:
        You need to first generate a count vector from the input data, excluding the NLTK
        stopwords. This can be done by importing the English stopword list from NLTK and then
        passing it to a CountVectorizer during initialization.

        CountVectorizer is slightly different than the FreqDist object you used in your previous
        assignment.  Where FreqDist is good at creating a dict-like bag-of-words representation for
        a single document, CountVectorizer is optimized for creating a sparse matrix representing
        the bag-of-words counts for every document in a corpus of documents all at once.  Both
        objects are useful at different times.

    How to create tf-idf features:
        tf-idf features can be computed using TfidfTransformer with the count matrix (BOWs matrix)
        as an input. The fit method is used to fit a tf-idf estimator to the data, and the
        transform method is used afterwards to transform either the training or test count-matrix
        to a tf-idf representation. The method fit_transform strings these two methods together
        into one.

        For the training data, you'll want to use the fit_transform method to both fit the
        tf-idf model and then transform the training count matrix into a tf-idf representation.

        For the test data, you only need to call the transform method since the tf-idf model
        will have already been fit on your training set.


        
    test_data : List[str]
        Test data in list
    
    min_docs : integer
        Do not include terms in the vocabulary that occur in less than "min_docs" documents 

    Returns
    -------
    Tuple(scipy.sparse.csr.csr_matrix,..)
        Returns X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf as a tuple.

    Examples
    --------
    >>> X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf =
    ...          extract_text_features(twenty_train.data, sample_test_documents, 1)
    >>> X_train_counts
    <2989x39831 sparse matrix of type '<class 'numpy.int64'>'
	with 377208 stored elements in Compressed Sparse Row format>
    """

    # Tokenize the text and store the result in 'X_train_counts'

    ### YOUR SOLUTION STARTS HERE### 

    # Generate a count vector from the input data, excluding the NLTK stopwords
    count_vec = CountVectorizer(stop_words=stopwords.words('english'), min_df = min_docs) 

    # Generate BOW and store in 'X_train_counts', and 'X_test_counts'
    X_train_counts = count_vec.fit_transform(train_data)
    X_test_counts = count_vec.transform(test_data)

    # Compute tfidf feature values and store in 'X_train_tfidf' and 'X_test_tfidf'
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return (X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf)
    ### END SOLUTIONS ###pass




#############
# Problem 3 #
#############

def fit_and_predict_multinomialNB(X_train, Y_train, X_test):

    """
    Train multinomial naive Bayes model with 'X_train' and 'Y_train' and
    predict the Y values for 'X_test'. (Use 'MultinomialNB' from scikit-learn.)
    Return the predicted Y values.

    Parameters
    ----------
    X_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    Y_train: numpy.ndarray
        Labels for training data (target value)
    X_test: scipy sparse matrix
        Test data used for prediction

    Returns
    -------
    numpy.ndarray[int]
        Target values predicted from 'X_test'

    Examples
    --------
    >>> predicted_multNB = fit_and_predict_multinomialNB(X_train_tfidf, twenty_train.target, X_test_tfidf)
    >>> print(predicted_multNB)
    [4 1]

    """

    # Import the package
    from sklearn.naive_bayes import MultinomialNB

    ### YOUR SOLUTION STARTS HERE### 

    # Train
    clf_multNB = MultinomialNB().fit(X_train, Y_train)
    # Predict
    predicted_multNB = clf_multNB.predict(X_test)
    return predicted_multNB
    ### END SOLUTIONS ###pass


def fit_and_predict_LR(X_train, Y_train, X_test):

    """
    Train logistic regression model with 'X_train' and 'Y_train' and
    predict the Y values for 'X_test'. (Use 'LogisticRegression' from scikit-learn.)
    Return the predicted Y values.


    Parameters
    ----------
    X_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    Y_train: numpy.ndarray
        Labels for training data (target value)
    X_test: scipy sparse matrix
        Test data used for prediction

    Returns
    -------
    numpy.ndarray[int]
        Target values predicted from 'X_test'

    """

    # Import the package
    from sklearn.linear_model import LogisticRegression

    ### YOUR SOLUTION STARTS HERE### 
    
    clfLR = LogisticRegression().fit(X_train, Y_train)
    # print('Intercept terms for logistic regression are:',clfLR.intercept_)
    # Predict
    predicted_LR = clfLR.predict(X_test)
    return predicted_LR
    ### END SOLUTIONS ###pass


def fit_and_predict_KNN(X_train, Y_train, X_test, K):

    """
    Train nearest neighbor classifier model with 'X_train' and 'Y_train' and
    predict the Y values for 'X_test'. Use 'KNearestNeighborsClassifier' from 
    scikit-learn with K nearest neighbors (K = 1, 3, 5, ....)
    Return the predicted Y values.


    Parameters
    ----------
    X_train: scipy sparse matrix
        Data for training (matrix with features, e.g. BOW or tf-idf)
    Y_train: numpy.ndarray
        Labels for training data (target value)
    X_test: scipy sparse matrix
        Test data used for prediction
    K: integer (odd)
    	Number of neighbors to use for prediction, e.g., K = 1, 3, 5, ...

    Returns
    -------
    numpy.ndarray[int]
        Target values predicted from 'X_test'

    """
 
    # Import the package
    from sklearn.neighbors import KNeighborsClassifier

    ### YOUR SOLUTION STARTS HERE###
    
    # fit the model (for KNN this is just storing the training data and labels)
    clfKNN = KNeighborsClassifier(n_neighbors=K).fit(X_train, Y_train) 
    # Predict
    predicted_KNN = clfKNN.predict(X_test)
    return predicted_KNN
    ### END SOLUTIONS ###pass
    



#############
# Problem 4 #
#############
def test_classifiers(train, test, min_docs, K):
    """
    Evaluate the accuracy of multiple classifiers by training on the data in 
    "train" and making predictions on the data in "test". The classifiers
    evaluated are: MultinomialNB, Logistic, and kNN.
    The input train and test data are scikit-learn objects of type "bunch"
    containing both the raw text as well as label information.
    The function first call extract_text_features() to create a common
    vocabulary and feature set for all the classifiers to use.
    
    
    Parameters
    ----------
    train: sklearn.datasets.base.Bunch
        Text data with labels for training each classifier
    test: sklearn.datasets.base.Bunch
        Text data with labels for testing each classifier
    min_docs : integer
        Do not include terms in the vocabulary that occur in less than "min_docs" documents    
    K: integer (odd)
        Number of neighbors to use for prediction, e.g., K = 1, 3, 5, ...
 
    """
    X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf = extract_text_features(train.data, test.data, min_docs)
    
    num_docs, vocab_size = X_train_counts.shape
    print('Number of documents = %d' %(num_docs))
    print('Vocabulary size = %d' %(vocab_size))
    
    predicted_multNB = fit_and_predict_multinomialNB(X_train_tfidf, train.target, X_test_tfidf)
    predicted_LR = fit_and_predict_LR(X_train_tfidf, train.target, X_test_tfidf) 
    predicted_KNN = fit_and_predict_KNN(X_train_tfidf, train.target, X_test_tfidf, K)
    
    
    # Now evaluate the classifiers on the test data
    # Print out the accuracy as a percentage for each classifier.
    # np.mean() is used to calculate the accuracy. Round the accuracy to 4 decimal places.
    import numpy as np
    multNB_percent_test_accuracy = 100*np.mean(predicted_multNB == test.target)
    print('Accuracy with multinomial naive Bayes: %4.2f'  % multNB_percent_test_accuracy) 
    
    LR_percent_test_accuracy = 100*np.mean(predicted_LR == test.target)
    print('Accuracy with logistic regression: %4.2f'  % LR_percent_test_accuracy) 
    
    KNN_percent_test_accuracy = 100*np.mean(predicted_KNN == test.target)
    print('Accuracy with kNN, k=%d, classifier: %4.2f'  % (K, KNN_percent_test_accuracy)) 
      

#############
# Problem 5 #
#############

def LR_weights_and_words(traindata, M):

    """
    Train a logistic classifier using the data in traindata and print
    out the M terms with the largest positive weights and the M terms with
    the largest negative weights (print both the strings for the terms and their weights)
    
    Use the same settings as used before in extract_text_features() to tokenize
    the set of documents, generate a vocabulary, extract a bag of words array,
    and to convert to tfidf representation. Then train a logistic regression
    classifier using the default settings.
     
    Parameters
    ----------
    traindata:  (sklearn.datasets.base.Bunch)
        an sklearn object for storing data sets (such as twenty_train) with
        documents stored in the "data" attribute and labels in the "target" attribute
    M: integer
        Number of positive weights and negative weights to print out
 
    """

    # Import the package
    from sklearn.linear_model import LogisticRegression

    ### YOUR SOLUTION STARTS HERE### 
    # Note: See notes from class lecture about mapping feature IDs to term names
    # Note: use the coef_ attribute of the LogisticRegression type from scikit_learn  

    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    X_train_counts = vectorizer.fit_transform(traindata.data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    logreg = LogisticRegression()
    logreg.fit(X_train_tfidf, traindata.target)

    featureIdx_to_words = vectorizer.get_feature_names()
                                                                                                                    
    for class_idx in range(logreg.coef_.shape[0]):
        # order features for class #class_idx
        print(traindata.target_names[class_idx])
        
        # grab weights for this class
        weights = logreg.coef_[class_idx,:]
        sorted_idxs = np.argsort(weights)
        min_weight_idxs = sorted_idxs[0:M]
        max_weight_idxs = sorted_idxs[-(M+1):-1]

        for idx in reversed(max_weight_idxs):
            print('%s:    %.4f' %(featureIdx_to_words[idx], logreg.coef_[class_idx, idx]))

        print("---------------------")

        for idx in reversed(min_weight_idxs):
            print('%s:    %.4f' %(featureIdx_to_words[idx], logreg.coef_[class_idx, idx]))
        
        print('\n')

 
    ### END SOLUTIONS ###pass
 


#############
# Problem 6 #
#############
def confusion_matrix(predicted_labels, target_labels):

    """
    Print to the screen an M x M confusion matrix, given as input two arrays
    of integers of the same length, where the first is the predicted labels
    and the second is the true labels (and the labels take values from 1 to M). 
    Entry (i,j) of the confusion matrix should show the number of items (documents)
    that were predicted to be in class i and are actually in class j.
    
    NOTE: write this function using basic Python functions, do not call
    other existing code that computes a confusion matrix
    
    Parameters
    ----------
    predicted_labels: numpy.ndarray
        N predicted labels, the labels taking values from 1 to M
    target_labels: numpy.ndarray
        N target labels, the labels taking values from 1 to M
        
 
    """
   
    ### YOUR SOLUTION STARTS HERE###  

    M = len(np.unique(target_labels))
    mat = np.zeros((M,M))

    for label_idx in range(len(target_labels)):
        mat[predicted_labels[label_idx], target_labels[label_idx]] += 1

    print(mat)

    ### END SOLUTIONS ###pass
    
   






#############
# Problem 7 #
#############
# Call the functions and compute various statistics and accuracies....
#if __name__ == '__main__':
print('\n\n---PART1---')
    
# Load the news dataset, and select only the top n_categories to work with
n_categories = 5
twenty_train, twenty_test = load_news_dataset(n_categories)

# # Play with the dataset
# # print the first few lines of the first document in the data set
print('The names of the %d most common labels in the data set are:' %(n_categories))
print(twenty_train.target_names)
print('\n')

len(twenty_train.data)
len(twenty_train.filenames)
print('First few lines of the first document....')
print('\n'.join(twenty_train.data[0].split('\n')[:3]))
print('\n')

# Print the labels (target names) of the first 10 documents
print('Labels (targets) of the first 10 documents...')
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
print('\n')
input('Hit return to continue')

#----------------------------------------------------------------------------#
print('\n\n---PART2---')
# Define sample test data  
sample_test_documents = [
    # should be classified as soc.religion.christian
    'God is love',
    # should be classified as rec.sport.baseball
    'The major league baseball player has a strong reputation'
]

# create TFIDF representations of the training data and sample test data
X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf = extract_text_features(twenty_train.data, sample_test_documents,1)

print('Dimensions of X_train_counts are (%d x %d)' %(X_train_counts.shape[0], X_train_counts.shape[1]))
print('\n')
num_non_zero = X_train_counts.nnz
av_num_word_tokens_per_doc = X_train_counts.sum(axis=1).mean()
av_num_docs_per_word_token = X_train_counts.sum(axis=0).mean()
num_docs, vocab_size = X_train_counts.shape
density = 100*num_non_zero/(num_docs*vocab_size)

print('\n')
print('Number of non-zero elements in X_train_counts: %d' %(num_non_zero))  # 372208
print('Percentage of non-zero elements in X_train_counts: %.2f' % (density))  # 0.31 percent
print('Average number of word tokens per document: %.2f' % (av_num_word_tokens_per_doc))  # 181.08
print('Average number of documents per word token: %.2f' % (av_num_docs_per_word_token))  # 13.59
print('\n')
input('Hit return to continue')


#----------------------------------------------------------------------------#
print('\n\n---PART3---') 
predicted_multNB = fit_and_predict_multinomialNB(X_train_tfidf, twenty_train.target, X_test_tfidf)
predicted_LR = fit_and_predict_LR(X_train_tfidf, twenty_train.target, X_test_tfidf)
K=3
predicted_KNN = fit_and_predict_KNN(X_train_tfidf, twenty_train.target, X_test_tfidf, K)

#
# # Print out the results to see how they are classified.
# # Multinomial naive Bayes
print('Predicted labels with multinomial NB classifier:')
for doc, category in zip(sample_test_documents, predicted_multNB):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
print('\n')

# kNN
print('Predicted labels with kNN classifier:')
for doc, category in zip(sample_test_documents, predicted_KNN):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
print('\n')
#
# Logistic Regression
print('Predicted labels with logistic classifier:')
for doc, category in zip(sample_test_documents, predicted_LR):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
print('\n')
#
# """
#     'God is love'  should be classified as soc.religion.christian
#     'The major league baseball player has a strong reputation' should be classified as rec.sport.baseball.
#     Are they all correctly classified? If not, discuss why. (not graded)
# """

input('Hit return to continue')




#----------------------------------------------------------------------------#
print('\n\n---PART4---')
# call the test_classifiers() function to test the accuracy of the classifiers 
# on the twenty_train and twenty_test data sets, for each of the following
# min_docs values: [1, 3, 5, 10, 100].  Use K=3 in all cases.
# (Note: this uses the twenty_test data set for testing, not sample data)
K=3
for mindocs in [1,3,5,10,100]:
    print('Evaluating classifiers with min_docs = %d' %(mindocs))
    test_classifiers(twenty_train, twenty_test, mindocs, K) 
    print('\n')

  
"""
Example:
test_classifiers(twenty_train, twenty_test, 1, 3)
Your output should be similar to that below...
	Accuracy with multinomial naive Bayes: 97.28
	Accuracy with logistic regression: 96.73  <- may vary from run to run due to some randomness in how LR models are fit
	Accuracy with KNN, k=3, classifier: 91.40
"""

input('Hit return to continue')




#----------------------------------------------------------------------------#
print('\n\n---PART5--')
# Now evaluate the accuracy of the KNN classifier for different values of K,
# with min_docs = 1
X_train_counts, X_train_tfidf, X_test_counts, X_test_tfidf = extract_text_features(twenty_train.data, twenty_test.data,1)
Kvalues = [1, 3, 5, 7, 9, 11]
for K in Kvalues:
	predicted_KNN = fit_and_predict_KNN(X_train_tfidf, twenty_train.target, X_test_tfidf, K)
	KNN_percent_test_accuracy = 100*np.mean(predicted_KNN == twenty_test.target)
	print('Accuracy with kNN classifier, k = %d: %4.2f'  % (K, KNN_percent_test_accuracy)) 


#----------------------------------------------------------------------------#                                                                                   
LR_weights_and_words(traindata=twenty_train, M=10)
confusion_matrix(predicted_KNN, twenty_test.target)
