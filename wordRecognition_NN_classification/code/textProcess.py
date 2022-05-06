import re
import pandas as pd
import numpy as np
import csv
import difflib

def openFile(filepath):
    with open(filepath,'r') as f:
        text = f.read()
    return text


def cut_text(text, lenth):
    textArr = re.findall('.{' + str(lenth) + '}', text)
    textArr.append(text[(len(textArr) * lenth):])
    print (type(textArr))
    return textArr

def createIndexList(sumIndex):
    indexList = []
    for i in range(sumIndex):
        indexList.append(i)
    indexStrList = [str(i) for i in indexList]
    return indexStrList

def text2csv(csvfile,name,newList):
    test = pd.DataFrame(columns=name, data=newList)  # 数据有三列，列名分别为one,two,three
    #print(test)
    test.to_csv(csvfile, index=False, header=True)

def indexSlice(indexfilePath,sumofIndex,lenofSeq=30):
    '''
    用来将整个训练或者测试数据按lenofSeq进行平均切片，得到行的第一个数字和最后一个数字，以
    ‘1 30
    31 60
    ...'
    的形式存储到文件中，去和训练数据得到的position进行比对，这样能得出每一行有没有需要查找出来的数据，从而给每一个seq一个标签 0 OR 1
    :param indexfilePath: 
    :param sumofIndex: 
    :param lenofSeq: 
    :return: 
    '''
    maxIndex = sumofIndex - sumofIndex%lenofSeq
    mage = list(range(1, maxIndex+1))
    # mage = [str(i) for i in mage]

    max_count = len(mage)
    n = 0

    filepath = indexfilePath
    findex = open(filepath, 'w+')
    while n < max_count:  # 这里用到了一个while循环，穿越过来的
        # print(mage[n:n+10]) #这里就用到了切片，这一节的重点

        #print("{0} {1}".format(mage[n], mage[n + 9]))

        print("{0} {1}".format(mage[n], mage[n+lenofSeq-1]), file=findex)
        n = n + lenofSeq
    findex.close()

def getPositivePosition(indexFile,positionFile):
    '''
    这个函数是用来取得放label=1的位置的
    通过比较index中的开头和结尾两个数是否包含真正position的位置，例如：
    index中：  position中：
    1 30      1 9
    31 60     67 78
    61 90
    那么得出来的位置的集合就是{1,3}即第一行和第三行具有label=1的标签
    :param indexFile: 
    :param positionFile: 
    :return: 
    '''

    def loadDataSet(filepath):
        '''
        将txt文件以多维list形式存储在list里面
        :param filepath: 
        :return: 
        '''
        file = open(filepath, "r")
        List_row = file.readlines()

        list_source = []
        for list_line in List_row:
            list_line = list(list_line.strip().split(' '))

            s = []
            for i in list_line:
                s.append(int(i))

            list_source.append(s)
        # print(list_source)

        return list_source

    resultListindex = loadDataSet(indexFile)
    resultpo = loadDataSet(positionFile)

    #print(resultpo)
    #print(resultListindex[0][1])

    #这里用set防止重复
    setcover = set()
    for i in range(len(resultListindex)):
        for j in range(len(resultpo)):
            if resultListindex[i][0] <= resultpo[j][0] and resultListindex[i][1] >= resultpo[j][1]:
                setcover.add(i)
    listLine = sorted(setcover)
    return listLine

def giveLabel(positionList,lines):
    labelList = []
    for i in range(lines):
        labelList.append(0)
    #print(labelList)

    for j in range(len(labelList)):
        for k in range(len(positionList)):
            if j == positionList[k]:
                labelList[j] = 1
    return labelList








if __name__ == '__main__':


    #处理较小的training数据集
    traintext = openFile('../rawData/trainingdata.txt')
    trainlist = cut_text(traintext,30)
    trainlist.pop()

    sumIndex = len(traintext)
    print (sumIndex)
    indexList = createIndexList(sumIndex)

    print(len(indexList))


    indexSlice('../rawData/trainingIndexPo.txt',sumIndex,30)

    listLine = getPositivePosition('../rawData/trainingIndexPo.txt','../rawData/trainingPosition.txt')

    #print(len(trainlist))
    labelList = giveLabel(listLine,len(trainlist))

    newList = list(zip(indexList,trainlist,labelList))

    text2csv('../data/train.csv',['id', 'sequence','label'],newList)

    ####################################################################

    '''
    #处理bigTraining数据集
    traintext = openFile('../rawData/bigTrainingdata.txt')
    trainlist = cut_text(traintext,30)
    trainlist.pop()

    sumIndex = len(traintext)
    print (sumIndex)
    indexList = createIndexList(sumIndex)

    print(len(indexList))


    indexSlice('../rawData/bigTrainingIndexPo.txt',sumIndex,30)

    listLine = getPositivePosition('../rawData/bigTrainingIndexPo.txt','../rawData/bigTrainingPosition.txt')

    #print(len(trainlist))
    labelList = giveLabel(listLine,len(trainlist))

    newList = list(zip(indexList,trainlist,labelList))

    text2csv('../data/bigTrain.csv',['id', 'sequence','label'],newList)
    '''
    ##############################################################

    '''
    #处理test数据集,得到真实的测试数据的标签
    traintext = openFile('../rawData/testdata.txt')
    trainlist = cut_text(traintext,30)
    trainlist.pop()

    sumIndex = len(traintext)
    print (sumIndex)
    indexList = createIndexList(sumIndex)

    print(len(indexList))


    indexSlice('../rawData/testIndexPo.txt',sumIndex,30)

    listLine = getPositivePosition('../rawData/testIndexPo.txt','../rawData/testPosition.txt')

    #print(len(trainlist))
    labelList = giveLabel(listLine,len(trainlist))

    newList = list(zip(indexList,trainlist,labelList))

    text2csv('../data/realtest.csv',['id', 'sequence','label'],newList)


    #得到需要测试的数据（即没有label的csv文件）
    testlist = list(zip(indexList,trainlist))
    text2csv('../data/test.csv',['id', 'sequence'],testlist)
    '''













