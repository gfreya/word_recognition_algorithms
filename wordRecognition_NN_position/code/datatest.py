

'''
numlist = []
for i in range(83869):
    numlist.append(i)



strNumlist = [str(i) for i in numlist]
print (strNumlist)
'''



'''

mage=list(range(1,101))
#mage = [str(i) for i in mage]

max_count=len(mage)              #使用len()获取列表的长度，上节学的
n=0

filepath = '../test1.txt'
findex = open(filepath, 'w+')
while n<max_count:               #这里用到了一个while循环，穿越过来的
    #print(mage[n:n+10]) #这里就用到了切片，这一节的重点

    print("{0} {1}".format(mage[n], mage[n + 9]))

    print("{0} {1}".format(mage[n],mage[n+9]),file=findex)
    n=n+10
findex.close()

'''

'''
filepath = '../test1.txt'

findex = open(filepath, 'w+')
for i in range(int(234/10)):

    print("{0} {1}".format(i,i+9),file=findex)
'''
# -*- coding:gbk -*-

'''
import pandas as pd

df = pd.read_csv('../rawData/trainingCleanWords.csv',encoding='gbk', index_col=1)

final_list = df.tolist()
print(final_list[1])
'''

'''
import numpy as np

nd = np.genfromtxt('../rawData/trainingCleanWords.csv', delimiter=',', skip_header=True)

final_list = nd.tolist()
'''
'''
import csv
import numpy as np
file_name = '../rawData/trainingCleanWords.csv'

f = open(file_name, 'r')
csvreader = csv.reader(f)
final_list = list(csvreader)
col_list = [x[1] for x in final_list]
print (len(col_list[1]))
'''
import string
import random

'''
list = ['car','chapter']

preList = []
tmp = string.ascii_letters.lower()
for i in range(len(list)):

    preList.insert(i,''.join(random.choice(tmp) for _ in range(random.randint(1, 30-len(list[i])))))

print(preList)

postList=[]
for i in range(len(list)):
    salt = ''.join(random.sample(tmp, 30-len(list[i])-len(preList[i])))
    postList.append(salt)

print(postList)
'''

'''
def addList(list1, list2,list3):
    addstr = []
    for i in range(len(list1)):
        addstr.append(list1[i]+list2[i]+list3[i])

    return addstr

finalWords = addList(preList,list,postList)
print(finalWords)

def countlen(rawList):
    output=[]
    for i in range(len(rawList)):
        output.append(len(rawList[i]))
    return output

lenList = countlen(list)
lenPreList = countlen(preList)


print(type(lenPreList[0]))
print(lenList)

def position(list1,list2):
    startLabel = []
    lastLable = []
    for i in range(len(list1)):
        startLabel.append(list1[i]+1)
        lastLable.append((list1[i]+list2[i]))
    return startLabel,lastLable

start,last = position(lenPreList,lenList)
print(start)
print(last)
'''
import numpy as np

lista = np.array([1,2,3,4,5,6,7,8,9,10,11,21,34,56,78])
listb=lista.reshape(5,3)
print(listb)
print (listb.shape[0])

result = []
for i in range(listb.shape[0]):
    result.append(np.argmax(listb[i][:])+1)
print (result)








