import pandas as pd
import numpy as np
import csv

def readFile_getCol(filePath,colname):
    with open(filePath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        column = [row[colname] for row in reader]
    return column

def get_trueIndex(preList,realList):
    trueIndex = []
    for i in range(len(preList)):
        if preList[i] == realList[i]:
            trueIndex.append(i)
    acc = len(trueIndex)/len(preList)
    return trueIndex,acc




if __name__ == "__main__":
    trainPreList = readFile_getCol('../result/testpre(train).csv', 'label')
    trainLastList = readFile_getCol('../result/testlast(train).csv','label')
    bigTrainPreList = readFile_getCol('../result/testpre(bigTrain).csv', 'label')
    bigTrainLastList = readFile_getCol('../result/testlast(bigTrain).csv', 'label')
    realPreList =readFile_getCol('../data/realtest.csv','startLabel')
    realLastList = readFile_getCol('../data/realtest.csv','lastLabel')

    truepreTrainIndex,trainpreAcc = get_trueIndex(trainPreList,realPreList)
    truelastTrainIndex,trainlastAcc = get_trueIndex(trainLastList, realPreList)
    trueprebigTrainIndex,bigTrainpreAcc = get_trueIndex(bigTrainPreList, realPreList)
    truelastbigTrainIndex,bigTrainlastAcc = get_trueIndex(bigTrainLastList, realPreList)

    print(trainpreAcc,trainlastAcc,bigTrainpreAcc,bigTrainlastAcc)






