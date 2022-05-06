#!/usr/bin/env python
# -*- coding: utf-8 -*-


import nltk

import jieba
import re
import os
from nltk.corpus import stopwords as pw
import string, random
from nltk.stem import SnowballStemmer
import scipy.misc as misc
import random
from skimage import io, data, transform
import numpy as np
import csv

def readFile_getCol(filePath,colname):
    with open(filePath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        column = [row[colname] for row in reader]
    return column

def newindex(labelList):
    newIndex=[]
    for i in range(len(labelList)):
        newIndex.append(i*30+int(labelList[i]))
    return newIndex

def position2txt(outfilepath,prelist,lastlist):
    findex = open(outfilepath, 'w+')
    for i in range(len(prelist)):
        print("{0} {1}".format(prelist[i], lastlist[i]), file=findex)
    findex.close()


if __name__ == '__main__':
    '''
    #处理train的数据

    trainsequence = readFile_getCol('../rawData/trainpre.csv','sequence')
    trainSequencePath = '../HMMdata/trainsequence.txt'
    findex = open(trainSequencePath, 'w+')
    print ("".join(trainsequence),file=findex)

    trainprelabel = readFile_getCol('../rawData/trainpre.csv','label')
    trainPre = newindex(trainprelabel)
    trainlastlabel = readFile_getCol('../rawData/trainlast.csv', 'label')
    trainLast = newindex(trainlastlabel)

    trainPositionPath = '../HMMdata/trainPosition.txt'
    position2txt(trainPositionPath,trainPre,trainLast)
    '''
    ########################################################################

    # 处理bigTrain的数据

    trainsequence = readFile_getCol('../rawData/bigTrainpre.csv', 'sequence')
    trainSequencePath = '../HMMdata/bigTrainsequence.txt'
    findex = open(trainSequencePath, 'w+')
    print("".join(trainsequence), file=findex)

    trainprelabel = readFile_getCol('../rawData/bigTrainpre.csv', 'label')
    trainPre = newindex(trainprelabel)
    trainlastlabel = readFile_getCol('../rawData/bigTrainlast.csv', 'label')
    trainLast = newindex(trainlastlabel)

    trainPositionPath = '../HMMdata/bigTrainPosition.txt'
    position2txt(trainPositionPath, trainPre, trainLast)

    ########################################################################
    '''
    # 处理test的数据

    trainsequence = readFile_getCol('../rawData/realtest.csv', 'sequence')
    trainSequencePath = '../HMMdata/testsequence.txt'
    findex = open(trainSequencePath, 'w+')
    print("".join(trainsequence), file=findex)

    trainprelabel = readFile_getCol('../rawData/realtest.csv', 'startLabel')
    trainPre = newindex(trainprelabel)
    trainlastlabel = readFile_getCol('../rawData/realtest.csv', 'lastLabel')
    trainLast = newindex(trainlastlabel)

    trainPositionPath = '../HMMdata/testPosition.txt'
    position2txt(trainPositionPath, trainPre, trainLast)
    '''





