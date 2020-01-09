import numpy as np
import pandas as pd
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
import time

reddit_train = pd.read_csv('reddit_train.csv')
reddit_test = pd.read_csv('reddit_test.csv')
redditWords = reddit_train.T.iloc[1].to_numpy()
redditLabels = reddit_train.T.iloc[-1].to_numpy()


def encodeType(reddit_train):
    typeList = list()
    typeVector = list()

    for i in range(0, len(reddit_train)):
        subreddit = reddit_train[i]
        if subreddit not in typeList:
            typeList.append(subreddit)

    encoded = list()
    index = 0
    for element in typeList:
        my_tuple = (index, element)
        index = index + 1
        encoded.append(my_tuple)


    for i in range(0, len(reddit_train)):
        subreddit = reddit_train[i]

        for k in range(0, len(encoded)):
            if encoded[k][1] == subreddit:
                typeVector.append(encoded[k][0])

    return typeVector, typeList

redditLabels, typeList = encodeType(redditLabels)
x_train, x_test, y_train, y_test = train_test_split(redditWords, redditLabels, test_size=0.015)

def startKBert(x_train, y_train, x_test, y_test, typeList):
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                          x_test=x_test, y_test=y_test,
                                                                          class_names=typeList,
                                                                          preprocess_mode='bert',
                                                                          maxlen=250,
                                                                          max_features=40000)
    model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)
    learner.fit_onecycle(2e-5, 4)
    learner.validate(val_data=(x_test, y_test), class_names=typeList)
    predictor = ktrain.get_predictor(learner.model, preproc)
    predictor.get_classes()
    predictor.save('D:\\Mcgill\\U3 fall\\COMP 551\\p2\\tryBert\\tmp\\my03_ktrain_predictor')

startKBert(x_train, y_train, x_test, y_test, typeList)

def useKBert(x_output):
    predictor = ktrain.load_predictor('D:\\Mcgill\\U3 fall\\COMP 551\\p2\\tryBert\\tmp\\my03_ktrain_predictor')
    y_output = list()
    for element in x_output:
        y_output.append(predictor.predict(element))

    pred = pd.DataFrame(y_output, columns=['Category'])
    fileName = 'D:\\Mcgill\\U3 fall\\COMP 551\\p2\\tryBert\\result\\KBert' + time.strftime("%Y%m%d%H", time.localtime()) + '.csv'
    pred.to_csv(fileName, index=True, index_label='Id', header=True)



x_output = reddit_test.T.iloc[1].to_numpy()
useKBert(x_output)