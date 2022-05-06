# -*- coding:gbk -*-

import re
import pandas as pd
import numpy as np
import csv
import difflib
import string
import random

def readcsv_getCleanwords(filepath):
    f = open(filepath, 'r')
    csvreader = csv.reader(f)
    final_list = list(csvreader)
    col_list = [x[1] for x in final_list]
    rawListLen = []
    indexList = []
    for i in range(len(col_list)):
        rawListLen.append(len(col_list[i]))
        indexList.append(i)

    return col_list,rawListLen,indexList

def createPreList(rawList,sumLen=30):
    preList = []
    tmp = string.ascii_letters.lower()
    for i in range(len(rawList)):
        preList.insert(i, ''.join(random.choice(tmp) for _ in range(random.randint(1, sumLen-1-len(rawList[i])))))

    preListLen = []
    for j in range(len(rawList)):
        preListLen.append(len(preList[j]))

    return preList,preListLen

def createPostList(rawList,preList,sumLen=30):
    postList = []
    tmp = string.ascii_letters.lower()

    for i in range(len(rawList)):
        postLen = sumLen - len(rawList[i]) - len(preList[i])
        add = ''.join(random.sample(tmp,postLen))
        postList.append(add)
    return postList

def concatSequence(preList,rawList,postList):
    finalList = []
    for i in range(len(rawList)):
        finalList.append(preList[i]+rawList[i]+postList[i])
    return finalList

def positionLable(preListLen,rawListLen):
    startLabel = []
    lastLabel = []
    for i in range(len(preListLen)):
        startLabel.append(preListLen[i]+1)
        lastLabel.append(preListLen[i]+rawListLen[i])
    return startLabel,lastLabel

def text2csv(csvfile,name,newList):
    test = pd.DataFrame(columns=name, data=newList)  # 数据有三列，列名分别为one,two,three
    #print(test)
    test.to_csv(csvfile, index=False, header=True)

if __name__ == '__main__':


    # basic training file
    cleanfilepath = '../rawData/trainingCleanWords.csv'
    rawList, rawListLen, indexList = readcsv_getCleanwords(cleanfilepath)
    # print(rawList[0:4])
    preList, preListLen = createPreList(rawList)
    # print(preList[0:4],len(preList[4]))
    postList = createPostList(rawList, preList)
    finalSequence = concatSequence(preList, rawList, postList)
    startLabel, lastLabel = positionLable(preListLen, rawListLen)
    #print(startLabel[1])
    finalList = list(zip(indexList, finalSequence, startLabel, lastLabel))
    saveTrainfilepath = '../data/train.csv'
    name = ['id', 'sequence', 'startLabel', 'lastLabel']
    text2csv(saveTrainfilepath, name, finalList)

    #####################################################################
    '''
    # big training data
    cleanfilepath = '../rawData/bigTrainingCleanWords.csv'
    rawList, rawListLen, indexList = readcsv_getCleanwords(cleanfilepath)
    # print(rawList[0:4])
    preList, preListLen = createPreList(rawList)
    # print(preList[0:4],len(preList[4]))
    postList = createPostList(rawList, preList)
    finalSequence = concatSequence(preList, rawList, postList)
    startLabel, lastLabel = positionLable(preListLen, rawListLen)
    finalList = list(zip(indexList, finalSequence, startLabel, lastLabel))
    saveTrainfilepath = '../data/bigTrain.csv'
    name = ['id', 'sequence', 'startLabel', 'lastLabel']
    text2csv(saveTrainfilepath, name, finalList)
    '''
    ####################################
    '''
    #get realTest.csv
    cleanfilepath = '../rawData/testCleanWords.csv'
    rawList, rawListLen, indexList = readcsv_getCleanwords(cleanfilepath)
    # print(rawList[0:4])
    preList, preListLen = createPreList(rawList)
    # print(preList[0:4],len(preList[4]))
    postList = createPostList(rawList, preList)
    finalSequence = concatSequence(preList, rawList, postList)
    startLabel, lastLabel = positionLable(preListLen, rawListLen)
    finalList = list(zip(indexList, finalSequence, startLabel, lastLabel))
    saveTrainfilepath = '../data/realtest.csv'
    name = ['id', 'sequence', 'startLabel', 'lastLabel']
    text2csv(saveTrainfilepath, name, finalList)
    saveTestfilepath = '../data/test.csv'
    newName = ['id','sequence']
    testList = list(zip(indexList, finalSequence))
    text2csv(saveTestfilepath,newName,testList)
    '''
























