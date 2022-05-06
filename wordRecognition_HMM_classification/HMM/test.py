'''
text = ["one","abkdhhg","two","wkojnxbsg","three","yhomnbgf"]
print (type(list))

lenList = []
for i in range(len(text)):
    lenList.append(len(text[i]))

print(lenList)

lenList.insert(0,1)
print(lenList)






firstIndexAddLen = []
lastIndexAddLen = []
filepath = '../position.txt'
findex = open(filepath, 'w+')

for j in range(len(lenList)-1):
    if j == 0:
        firstIndexAddLen.append(1)
    else:
        firstIndexAddLen.append(firstIndexAddLen[j-1]+lenList[j])

    lastIndexAddLen.append(firstIndexAddLen[j] + lenList[j + 1] - 1)

    #print("{0} {1}".format(firstIndexAddLen[j], lastIndexAddLen[j]), file=findex)
#findex.close()
    #print (j)

newfirstIndex = firstIndexAddLen[::2]


print(firstIndexAddLen)
print(newfirstIndex)
print(lastIndexAddLen)

'''

'''
f=open('out.txt','r')
lines=f.readlines()
f.close()
n=0
for line in lines:
    n = n + 1
print(n)
'''

fa = open('../HMMdata/testPosition.txt')
a = fa.readlines()
fa.close()
fb = open('out.txt')
b = fb.readlines()
fb.close()
c = [i for i in a if i in b]
fc = open('C.txt', 'w')
fc.writelines(c)
fc.close()



