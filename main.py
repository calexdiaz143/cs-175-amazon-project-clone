# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix, hstack, save_npz, load_npz
from random import shuffle
import parsers as ps
import numpy as np
import json

LOADFILE = True
category_names = ['Musical_Instruments','Patio_Lawn_and_Garden']


def load_reviews(filename, percent):
    file_content = open(filename).read().split('\n')[:-1]
    parsedData = []
    
    for i in file_content:
        currItem = json.loads(i)
        newDict = dict()
        newDict['reviewerID'] = ps.idToNum(currItem['reviewerID'])
        newDict['overall'] = ps.overallToNum(currItem['overall'])
        newDict['helpful'] = ps.helpfulToNum(currItem['helpful'])
        newDict['unixReviewTime'] = currItem['unixReviewTime']
        newDict['asin'] = ps.idToNum(currItem['asin'])
        newDict['summary'] = currItem['summary']
        newDict['reviewText'] = currItem['reviewText']
        
        newList = [newDict[i] for i in newDict]
        parsedData.append(newList)

    splitIndex = int(percent*len(file_content))
    shuffle(parsedData)
    train = np.array(parsedData[:splitIndex])
    test = np.array(parsedData[splitIndex:])
    return train, test

def load_all():
    allReviewsTrain = []
    allReviewsTest = []
    trainCategories = []
    testCategories = []
    for name in category_names:
        currTrain, currTest = load_reviews('db/reviews_'+name+'_5.json',.75)
        trainCategories.append([name] * currTrain.shape[0])
        testCategories.append([name] * currTest.shape[0])
        allReviewsTrain.append(currTrain)
        allReviewsTest.append(currTest)

    allReviewsTrain = np.concatenate(allReviewsTrain)
    allReviewsTest = np.concatenate(allReviewsTest)
    trainCategories = np.concatenate(trainCategories)
    testCategories = np.concatenate(testCategories)

    train = list(zip(allReviewsTrain,trainCategories))
    test = list(zip(allReviewsTest,testCategories))

    shuffle(train)
    shuffle(test)

    allReviewsTrain, trainCategories = zip(*train)
    allReviewsTest, testCategories = zip(*test)

    return allReviewsTrain, allReviewsTest, np.array(trainCategories), np.array(testCategories)

def parse_bow(features):
    summaryCorpus = []
    reviewCorpus = []
    finalFeatures = []

    for i, x in enumerate(features):
        summaryCorpus.append(x[5])
        reviewCorpus.append(x[6])
        finalFeatures.append(x[0:5])

    summaryBow = ps.sparsify(summaryCorpus)
    reviewBow = ps.sparsify(reviewCorpus)
    finalFeatures = hstack([csr_matrix(finalFeatures, dtype=np.int64), summaryBow,reviewBow])
    return finalFeatures

def trainNaiveBayes(allReviewsTrain):
    X, Y = zip(*allReviewsTrain)
    classifier  = MultinomialNB().fit(X,Y)
    return classifier

if(LOADFILE):
    trainX = load_npz('trainX.npz')
    testX = load_npz('testX.npz')
    trainY = np.load('trainY.npy')
    testY = np.load('testY.npy')
else:
    trainX, testX, trainY, testY = load_all()
    trainX = parse_bow(trainX)
    testX = parse_bow(testX)
    save_npz('trainX',trainX)
    save_npz('testX',testX)
    np.save('trainY',trainY)
    np.save('testY',trainY)
    
#classifier = trainNaiveBayes(train)