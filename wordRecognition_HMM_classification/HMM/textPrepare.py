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

nltk.download('punkt')


class Prepare:
    def fileread(self, filepath):  # read raw materials
        f = open(filepath)
        raw = f.read()
        print('read raw file......')
        return raw

    def sentoken(self, raw):  # tokenize sentence
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_tokenizer.tokenize(raw)
        print('tokenize the sentences in the raw file...')
        return sents

    def cleanlines(self, line):  # delete punctuation and other signs
        p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
        p2 = re.compile(r'[(][: @ . , ？！\s][)]')
        p3 = re.compile(r'[「『]')
        p4 = re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \-\ \[\ \]\ ]')
        line = p1.sub(r' ', str(line))
        line = p2.sub(r' ', str(line))
        line = p3.sub(r' ', str(line))
        line = p4.sub(r' ', str(line))
        print('delete signs...')
        return line

    def wordtoken(self, sent):  # tokenize the words
        segWords = jieba.cut(sent, cut_all=False)
        print('tokenize the words...')
        return segWords

    def cleanwords(self, words):  # delete stop words


        stopwords = {}.fromkeys([line.strip() for line in open("E:\\UNet\\stopwords.txt")])
        cleanwords = ''
        for word in words:
            word = word.lower()
            if word not in stopwords:
                cleanwords += word
        print('delete the stop words and lower words...')
        return cleanwords

    def addList(self, list1, list2):
        addList = []
        for i in range(len(list1)):
            addList.append(list1[i])
            addList.append(list2[i])
        return addList

    '''
    def stemwords(self,cleanwordslist):    #词干提取
        temp=[]
        stemwords=[]
        stemmer=SnowballStemmer('english')
        porter=nltk.PorterStemmer()
        for words in cleanwordslist:
            temp+=[[stemmer.stem(w) for w in words]]
        for words in temp:
            stemwords+=[[porter.stem(w) for w in words]]
        return stemwords


    def wordstostring(self,stemwords):
        strline=[]
        for words in stemwords:
            strline += [w for w in words]
        return strline


    def main(self,raw,out_url,i):
        re_out=open(out_url,'a')
        sents=self.sentoken(raw)
        cleanline=[self.cleanlines(sent) for sent in sents]
        words=[self.wordtoken(cl) for cl in cleanline]
        cleanwords=self.cleanwords(words)
        stemwords=self.stemwords(cleanwords)
        strline=self.wordstostring(stemwords)
        re_out.write(str(i)+'\t')
        out_str=','.join(strline)
        re_out.write(out_str)
        re_out.write('\n')
        re_out.close()
    '''


if __name__ == "__main__":
    precessedTest = Prepare()

    flag = int(input("Enter your choice:"))

    if flag == 1:#处理training data
        # basic processes of raw text

        ##################################################
        #filePathRaw = "../rawdata/rawTrainingData.txt"

        #下面处理一个大一点的训练数据，作为对照组
        filePathRaw = "../rawdata/rawBigTrainingData.txt"

        raw = precessedTest.fileread(filePathRaw)
        sentences = precessedTest.sentoken(raw)
        lines = precessedTest.cleanlines(sentences)
        words = precessedTest.wordtoken(lines)

        cleanwords = precessedTest.cleanwords(words)
        cleanwords = cleanwords.split(' ')

        # delete all null in the list
        for i in range(cleanwords.count('')):
            cleanwords.remove('')

        # create an insert list
        # 目的：在每两个单词之间插入长度0-100随机长度的a-z随机字母的字符
        # 所以以下步骤是为了创建一个和原单词序列相同个数的随机数组
        tmp = string.ascii_letters.lower()
        addList = []
        for i in range(len(cleanwords)):
            addList.insert(i, ''.join(random.choice(tmp) for _ in range(random.randint(1, 100))))
            # 本来用的是下一行代码，但是到100的长度会报错，超过population or is negative只能到50，于是改成上面的代码
            # addList.insert(i,''.join(random.sample(tmp, random.randint(1, 50))))

        # print (len((addList)))

        # print(type(addList))

        '''
        # create a dict to add a random string to one string in cleanwords
        # 利用字典 将两个list结合，这样可以插空变成一个list
        insertDict = dict(zip(cleanwords,addList))
        print (len(insertDict))
        # new list for each random string in every two raw strings
        [cleanwords.insert(cleanwords.index(x) + 1, insertDict[x]) for x in insertDict]
        '''

        '''
        #将两个list合并（可以写成def函数）
        finalWords = []
        for i in range(len(cleanwords)):
            finalWords.append(cleanwords[i])
            finalWords.append(addList[i])
        '''

        finalWords = precessedTest.addList(cleanwords, addList)
        '''
        filePathList = 'E:/UNet/testdata(list).txt'
        flist = open(filePathList, 'w+')
        # print(type(cleanwords))
        print(finalWords, file=flist)
        flist.close()
        '''
        print(finalWords[0])
        print(len(finalWords))

        lenList = []
        for i in range(len(finalWords)):
            lenList.append(len(finalWords[i]))

        # print(lenList)

        lenList.insert(0, 1)
        # print(lenList)
        firstIndexAddLen = []
        lastIndexAddLen = []

        # 这里是把每个类似单词的组成部分的第一个字母和最后一个字母形成两个列表
        for j in range(len(lenList) - 1):
            if j == 0:
                firstIndexAddLen.append(1)
            else:
                firstIndexAddLen.append(firstIndexAddLen[j - 1] + lenList[j])

            lastIndexAddLen.append(firstIndexAddLen[j] + lenList[j + 1] - 1)

        # 这里的firstIndex和lastIndex两个列表用来存储仅仅是单词（即我们需要预测的单词的首字母和最后一个字母形成的列表。这里相当于是取了上面两个列表的奇数位置的数
        firstIndex = firstIndexAddLen[::2]
        lastIndex = lastIndexAddLen[::2]


        ############################################################
        #filepath = '../HMMdata/trainingPosition.txt'

        #对照组：
        filepath = '../HMMdata/bigTrainingPosition.txt'

        findex = open(filepath, 'w+')
        for j in range(len(firstIndex)):
            print("{0} {1}".format(firstIndex[j], lastIndex[j]), file=findex)
        findex.close()

        # 这个步骤先不急
        # ****************************************************************
        # 得到testdata(final)
        # print (type(cleanwords))
        # print('-----')

        # new text
        # newdata是将list变成str类型，这样便于输入写入到final文件中
        newdata = ''.join(finalWords)
        # print (newdata)

        # print(type(newdata))


        ##############################################################3
        #filePathFinal = '../HMMdata/trainingdata.txt'
        #对照组
        filePathFinal = '../HMMdata/bigTrainingdata.txt'

        ffinal = open(filePathFinal, 'w+')
        print(len(newdata))
        print(newdata, file=ffinal)
        ffinal.close()

    elif(flag==2):#处理test文件
        # basic processes of raw text
        filePathRaw = "../rawdata/rawTestData.txt"
        raw = precessedTest.fileread(filePathRaw)
        sentences = precessedTest.sentoken(raw)
        lines = precessedTest.cleanlines(sentences)
        words = precessedTest.wordtoken(lines)

        cleanwords = precessedTest.cleanwords(words)
        cleanwords = cleanwords.split(' ')

        # delete all null in the list
        for i in range(cleanwords.count('')):
            cleanwords.remove('')

        # create an insert list
        # 目的：在每两个单词之间插入长度0-100随机长度的a-z随机字母的字符
        # 所以以下步骤是为了创建一个和原单词序列相同个数的随机数组
        tmp = string.ascii_letters.lower()
        addList = []
        for i in range(len(cleanwords)):
            addList.insert(i, ''.join(random.choice(tmp) for _ in range(random.randint(1, 100))))
            # 本来用的是下一行代码，但是到100的长度会报错，超过population or is negative只能到50，于是改成上面的代码
            # addList.insert(i,''.join(random.sample(tmp, random.randint(1, 50))))

        # print (len((addList)))

        # print(type(addList))

        '''
        # create a dict to add a random string to one string in cleanwords
        # 利用字典 将两个list结合，这样可以插空变成一个list
        insertDict = dict(zip(cleanwords,addList))
        print (len(insertDict))
        # new list for each random string in every two raw strings
        [cleanwords.insert(cleanwords.index(x) + 1, insertDict[x]) for x in insertDict]
        '''

        '''
        #将两个list合并（可以写成def函数）
        finalWords = []
        for i in range(len(cleanwords)):
            finalWords.append(cleanwords[i])
            finalWords.append(addList[i])
        '''

        finalWords = precessedTest.addList(cleanwords, addList)
        '''
        filePathList = 'E:/UNet/testdata(list).txt'
        flist = open(filePathList, 'w+')
        # print(type(cleanwords))
        print(finalWords, file=flist)
        flist.close()
        '''
        print(finalWords[0])
        print(len(finalWords))

        lenList = []
        for i in range(len(finalWords)):
            lenList.append(len(finalWords[i]))

        # print(lenList)

        lenList.insert(0, 1)
        # print(lenList)
        firstIndexAddLen = []
        lastIndexAddLen = []

        # 这里是把每个类似单词的组成部分的第一个字母和最后一个字母形成两个列表
        for j in range(len(lenList) - 1):
            if j == 0:
                firstIndexAddLen.append(1)
            else:
                firstIndexAddLen.append(firstIndexAddLen[j - 1] + lenList[j])

            lastIndexAddLen.append(firstIndexAddLen[j] + lenList[j + 1] - 1)

        # 这里的firstIndex和lastIndex两个列表用来存储仅仅是单词（即我们需要预测的单词的首字母和最后一个字母形成的列表。这里相当于是取了上面两个列表的奇数位置的数
        firstIndex = firstIndexAddLen[::2]
        lastIndex = lastIndexAddLen[::2]

        filepath = '../HMMdata/testPosition.txt'
        findex = open(filepath, 'w+')
        for j in range(len(firstIndex)):
            print("{0} {1}".format(firstIndex[j], lastIndex[j]), file=findex)
        findex.close()

        # 这个步骤先不急
        # ****************************************************************
        # 得到testdata(final)
        # print (type(cleanwords))
        # print('-----')

        # new text
        # newdata是将list变成str类型，这样便于输入写入到final文件中
        newdata = ''.join(finalWords)
        # print (newdata)

        # print(type(newdata))
        filePathFinal = '../HMMdata/testdata.txt'
        ffinal = open(filePathFinal, 'w+')
        print(len(newdata))
        print(newdata, file=ffinal)
        ffinal.close()


