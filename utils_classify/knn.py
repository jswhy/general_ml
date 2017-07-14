#coding=utf-8
import numpy
import matplotlib.pyplot as plot
import math
import operator
'''
knn分类算法
inX：待分类的向量，行向量，ndarray
dataMatInput:标注好的数据集
labelsX：标签集
'''
def classfy(inX,dataMat,labelX,k=6):
    dataMatSize=dataMat.shape[0]
    diffMat=math.tile(inX,(dataMatSize,1))-dataMat
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=plot.labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems())
    key=operator.itemgetter(1, reverse=True)
    return sortedClassCount[0][0]
'''
asd
'''
def load_data(x0=1.0):
    dataMat,labelMat = [],[];
    file = open('readTxt.txt')
    for line in file.readlines():
        lineArr = line.strip().split()
        # lineArr = line.strip().split(',')
        dataMat.append([x0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))


a=numpy.array([[1,2,3],[2,3,4],[1,4,3]])
print(a)
print(a.sum(axis=1))