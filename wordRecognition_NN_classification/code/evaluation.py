import pandas as pd
import numpy as np
import csv

def readFile_getCol(filePath,colname):
    with open(filePath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        column = [row[colname] for row in reader]


    return column

def get_accuracy(preList,realList):
    TP = []
    TN = []
    FP = []
    FN = []

    for i in range(len(preList)):
        if preList[i] == '1' and realList[i] == '1':
            TP.append(i)
        elif preList[i] == '0' and realList[i] == '0':
            TN.append(i)
        elif preList[i] == '1' and realList[i] == '0':
            FP.append(i)
        else:
            FN.append(i)

    accuracy = (len(TP)+len(TN))/len(preList)
    return accuracy


if __name__ == "__main__":
    preList = readFile_getCol('../result/resultpredicttest(small).csv', 'prediction')
    realList = readFile_getCol('../data/realtest.csv', 'label')
    accuracy = get_accuracy(preList,realList)
    print(accuracy)



