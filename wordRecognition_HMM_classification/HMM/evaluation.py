import difflib
import sys

def compMethodOne(realFile,predictFile,evaloutputfile):
    '''
    比较预测的位置和正确的位置
    :param realFile:
    :param predictFile:
    :param evaloutputfile:
    :return:
    '''
    realResult = open(realFile,'U').readlines()
    predictResult = open(predictFile,'U').readlines()

    diff = difflib.HtmlDiff()
    with open(evaloutputfile,'w') as f:
        for line in diff.make_file(realResult,predictResult):
            f.write(line)
    f.close()


'''
def compMethodTwo(realFile,predictFile,evaloutputfile):
    realResult = open(realFile)
    predictResult = open(predictFile)
    row = 0
    outfile = open(evaloutputfile,'w')
    for liner,linep in zip(realResult,predictResult):
        row += 1
        if not liner == linep:
            col = 0
            for charr,charp in zip(liner,linep):
                col += 1
                if not charr == charp:
                    print("difference in row:%d col%d\n"%(row,col))
                    outfile.write("difference in row:%d col%d"%(row,col))
                    break
'''

def comMethodTwo(realFile,predictFile,evaloutputfile):
    '''
    #这个diff的输出只能在控制台无法写入文件,diff的type是generator
    #所以用''.join()方法将其变成str类型，以便写入文件
    #记得最后一定要写上f.close()不然文件写入是空白
    :param realFile: 
    :param predictFile: 
    :return: 
    '''
    outfile = open(evaloutputfile,'w')
    diff = difflib.ndiff(open(realFile).readlines(), open(predictFile).readlines())
    compareResult = ''.join(diff)
    print(compareResult)
    #print(type(diff))
    #print(type(compareResult))
    #print(''.join(diff),file=outfile)
    outfile.write(compareResult)
    outfile.close()

def countLines(inputFile):
    f = open(inputFile, 'r')
    lines = f.readlines()
    f.close()
    n = 0
    for line in lines:
        n = n + 1
    return n

def getSameLine(realFile,predictFile,outSameFile):
    fr = open(realFile)
    frline = fr.readlines()
    fr.close()
    fp = open(predictFile)
    fpline = fp.readlines()
    fp.close()
    sameline = [i for i in frline if i in fpline]
    fs = open(outSameFile, 'w')
    fs.writelines(sameline)
    fs.close()

def countAuc(liner,lines):
    auc = lines/liner
    return auc



if __name__ =='__main__':

    #得到textdata的字符串个数
    with open('../HMMdata/testdata.txt') as f:
        data = f.read()
    allSample = len(data)
    print (allSample)

    '''
    #处理out.txt文件时的代码
    
    getSameLine('../HMMdata/testPosition.txt', 'out.txt', '../compareResult/sameLine.txt')

    liner = countLines('../HMMdata/testPosition.txt')
    lines = countLines('../compareResult/sameLine.txt')
    lineo = countLines('out.txt')


    #这里的completeauc指的是完全预测正确的条数，和下面的accuracy不一样
    completeauc = countAuc(liner,lines)
    filePathFinal = '../compareResult/outCompleteAuc.txt'
    ffinal = open(filePathFinal, 'w+')
    print(completeauc, file=ffinal)
    ffinal.close()
    print(completeauc)

    #accuracy指的是预测出来的条数与原来的比，只是看某个位置周围能不能预测出来有该单词，不该有单词的地方是不是也没有单词预测出来，并不需要全部一样，是为了与NN中的预测出来的label进行比较，因为NN中，只是做了分类问题，看能不能看出某一段sequence包不包含某个单词就行

    #为了和NN进行比较，还要知道TN的个数，其实就是本身没有，其实也没有的个数
    #整体个数用整个字符串的个数除以30得出，因为NN中是以30的长度切割的
    sequenceLen = 30
    allLine = int(allSample/30)
    accuracy = (allLine-liner+lineo)/allLine
    #print (accuracy)
    filePathFinal = '../compareResult/outAccuracy.txt'
    ffinal = open(filePathFinal, 'w+')
    print(accuracy, file=ffinal)
    ffinal.close()
    print(accuracy)


    flag = int(input("Enter your choice:"))

    if flag == 1: #将比较结果以html格式输出
        compMethodOne('../HMMdata/testPosition.txt', 'out.txt', '../compareResult/compMethodOne.html')
    elif flag == 2: #将比较结果输出到txt文件中
        comMethodTwo('../HMMdata/testPosition.txt', 'out.txt', '../compareResult/compMethodTwo.txt')
    else:
        print ("Wrong option!!!")
    

    '''
    #处理bigOut.txt
    getSameLine('../HMMdata/testPosition.txt', 'bigOut.txt', '../compareResult/bigsameLine.txt')

    liner = countLines('../HMMdata/testPosition.txt')
    lines = countLines('../compareResult/bigsameLine.txt')
    lineo = countLines('bigOut.txt')

    #这里的completeauc指的是完全预测正确的条数，和下面的accuracy不一样
    completeauc = countAuc(liner,lines)
    filePathFinal = '../compareResult/bigOutCompleteAuc.txt'
    ffinal = open(filePathFinal, 'w+')
    print(completeauc, file=ffinal)
    ffinal.close()
    print(completeauc)

    #accuracy指的是预测出来的条数与原来的比，只是看某个位置周围能不能预测出来有该单词，并不需要全部一样，是为了与NN中的预测出来的label进行比较，因为NN中，只是做了分类问题，看能不能看出某一段sequence包不包含某个单词就行

    # 为了和NN进行比较，还要知道TN的个数，其实就是本身没有，其实也没有的个数
    # 整体个数用整个字符串的个数除以30得出，因为NN中是以30的长度切割的
    sequenceLen = 30
    allLine = int(allSample / 30)
    print(allLine)
    accuracy = (allLine - liner + lineo) / allLine

    filePathFinal = '../compareResult/bigOutAccuracy.txt'
    ffinal = open(filePathFinal, 'w+')
    print(accuracy, file=ffinal)
    ffinal.close()
    print(accuracy)



    flag = int(input("Enter your choice:"))

    if flag == 1:  # 将比较结果以html格式输出
        compMethodOne('../HMMdata/testPosition.txt', 'out.txt', '../compareResult/bigcompMethodOne.html')
    elif flag == 2:  # 将比较结果输出到txt文件中
        comMethodTwo('../HMMdata/testPosition.txt', 'out.txt', '../compareResult/bigcompMethodTwo.txt')
    else:
        print("Wrong option!!!")









