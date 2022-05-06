#!/usr/bin/env python
# -*- coding:gbk -*-


import nltk

import jieba
import re
import pandas as pd
import csv
import os
from nltk.corpus import stopwords as pw
import string, random
from nltk.stem import SnowballStemmer
import scipy.misc as misc
import random
from skimage import io,data,transform
import numpy as np
nltk.download('punkt')

class Prepare:
    def fileread(self,filepath):  #read raw materials
        f = open(filepath)
        raw = f.read()
        print('read raw file......')
        return raw

    def sentoken(self,raw):       #tokenize sentence
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_tokenizer.tokenize(raw)
        print('tokenize the sentences in the raw file...')
        return sents

    def cleanlines(self,line):    #delete punctuation and other signs
        p1=re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
        p2=re.compile(r'[(][: @ . , ？！\s][)]')
        p3=re.compile(r'[「『]')
        p4=re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \-\ \[\ \]\ ]')
        line=p1.sub(r' ',str(line))
        line=p2.sub(r' ',str(line))
        line=p3.sub(r' ',str(line))
        line=p4.sub(r' ',str(line))
        print ('delete signs...')
        return line

    def wordtoken(self,sent):    #tokenize the words
        segWords = jieba.cut(sent, cut_all=False)
        print ('tokenize the words...')
        return segWords



    def cleanwords(self,words):   #delete stop words


        stopwords = {}.fromkeys([line.strip() for line in open("E:\\UNet\\stopwords.txt")])
        cleanwords = ''
        for word in words:
            word = word.lower()
            if word not in stopwords:
                cleanwords += word
        print('delete the stop words and lower words...')
        return cleanwords

    def addList(self,list1,list2):
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

    '''
    # basic processes of raw text
    filePathRaw = "../rawData/trainingdata.txt"
    raw = precessedTest.fileread(filePathRaw)
    sentences = precessedTest.sentoken(raw)
    lines = precessedTest.cleanlines(sentences)
    words = precessedTest.wordtoken(lines)

    cleanwords = precessedTest.cleanwords(words)
    cleanwords = cleanwords.split(' ')


    # delete all null in the list
    for i in range(cleanwords.count('')):
        cleanwords.remove('')
    name = ['cleanwords']
    test = pd.DataFrame(columns=name,data=cleanwords)
    test.to_csv('../rawData/trainingCleanWords.csv',encoding='gbk')
    '''
    ######################################################
    '''
    #train big training data
    filePathRaw = "../rawData/bigTrainingdata.txt"
    raw = precessedTest.fileread(filePathRaw)
    sentences = precessedTest.sentoken(raw)
    lines = precessedTest.cleanlines(sentences)
    words = precessedTest.wordtoken(lines)

    cleanwords = precessedTest.cleanwords(words)
    cleanwords = cleanwords.split(' ')

    # delete all null in the list
    for i in range(cleanwords.count('')):
        cleanwords.remove('')
    name = ['cleanwords']
    test = pd.DataFrame(columns=name, data=cleanwords)
    test.to_csv('../rawData/bigTrainingCleanWords.csv', encoding='gbk')
    '''
    ################################################
    #train test data
    filePathRaw = "../rawData/test.txt"
    raw = precessedTest.fileread(filePathRaw)
    sentences = precessedTest.sentoken(raw)
    lines = precessedTest.cleanlines(sentences)
    words = precessedTest.wordtoken(lines)

    cleanwords = precessedTest.cleanwords(words)
    cleanwords = cleanwords.split(' ')

    # delete all null in the list
    for i in range(cleanwords.count('')):
        cleanwords.remove('')
    name = ['cleanwords']
    test = pd.DataFrame(columns=name, data=cleanwords)
    test.to_csv('../rawData/testCleanWords.csv', encoding='gbk')










































