# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix, vstack, hstack
from random import shuffle
import parsers as ps
import numpy as np
import json

int('A2SUAM1J3GNN3B', 36)
def base36ToInt(base36):
    z = 0;
    mapping = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for i, c in enumerate(reversed(list(base36.lower()))):
        z += mapping.index(c) * 36 ** i
    return z
base36ToInt('A2SUAM1J3GNN3B')

def load_reviews(filename, category, percent):
    file_content = open(filename).read().split('\n')[:-1]
    parsedData = []
    
    for i in file_content:
<<<<<<< Updated upstream
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
        
=======
        tempDict = json.loads(i)
        tempDict['helpful'] = tempDict['helpful'][0]/(tempDict['helpful'][1] if tempDict['helpful'][1] != 0 else 1)

        # tempList = [tempDict[i] for i in tempDict]
        tempList = [tempDict['helpful']]
        parsedData.append((tempList, category))

    shuffle(parsedData)
    splitIndex = int(percent*len(file_content))
>>>>>>> Stashed changes
    print(parsedData[0])
    splitIndex = int(percent*len(file_content))    
    shuffle(parsedData)
    return np.array(parsedData[:splitIndex]), np.array(parsedData[splitIndex:])

# Loading Data
def load_all():
    musicalInstrumentTrain, musicalInstrumentTest = load_reviews('db/reviews_Musical_Instruments_5.json', 'music', .75)
    patioLawnGardenTrain, patioLawnGardenTest = load_reviews('db/reviews_Patio_Lawn_and_Garden_5.json', 'patio', .75)
    allReviewsTrain = np.concatenate((musicalInstrumentTrain, patioLawnGardenTrain))
    allReviewsTest = np.concatenate((musicalInstrumentTest, patioLawnGardenTest))
    shuffle(allReviewsTrain)
    shuffle(allReviewsTest)
    return allReviewsTrain, allReviewsTest

'''
    parsedData = [(featureList,category)...(featureList,category)]
    featureList = [f1,f2,f3(sparse-matrix),f4(sparse-matrix)]
'''
def trainNaiveBayes(allReviewsTrain):
    X, Y = zip(*allReviewsTrain)
    classifier  = MultinomialNB().fit(X,Y)
    return classifier


train, test = load_all()
<<<<<<< Updated upstream
classifier = trainNaiveBayes(train)
=======
print(train.shape)
print(test.shape)
trainNaiveBayes(train)
>>>>>>> Stashed changes
