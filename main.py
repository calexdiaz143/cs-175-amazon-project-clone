# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk import FreqDist
from nltk.corpus import stopwords
import numpy as np
import json
from random import shuffle

def load_reviews(filename, category, percent):
    file_content = open(filename).read().split('\n')[:-1] 
    parsedData = []
    for i in file_content:
        tempDict = json.loads(i)
        tempDict['helpful'] = tempDict['helpful'][0]/(tempDict['helpful'][1] if tempDict['helpful'][1] != 0 else 1)
        
        # tempList = [tempDict[i] for i in tempDict]
        tempList = [tempDict['helpful']]
        parsedData.append((tempList, category))
        
    shuffle(parsedData)
    splitIndex = int(percent*len(file_content))
    print(parsedData[0])
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

def trainNaiveBayes(allReviewsTrain):
    X, Y = zip(*allReviewsTrain)
    classifier  = MultinomialNB().fit(X,Y)
    return classifier


train, test = load_all()
print(train.shape)
print(test.shape)
trainNaiveBayes(train)