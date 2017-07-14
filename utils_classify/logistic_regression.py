# coding=utf-8
import math
import numpy as np
import random



'''
初始化函数，用来读取文件，并将文件拆分为数据集和标签集
'''
def load_data(filaname='readTxt.txt',x0=1.0):
    dataMat,labelMat = [],[];
    file = open(filaname)
    for line in file.readlines():
        # lineArr = line.strip().split()
        lineArr = line.strip().split(',')
        # lineArr = line.strip().split('/t')
        dataMat.append([x0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

def sigmoid(input):
    if type(input)==np.array or type(input)==np.matrix:
        if type(input)==np.array:
            np.mat(input)
        for i in input:
            i=1.0/(1+math.exp(-1*i))
        return  input
    else:
        return 1.0/(1+math.exp(-1*input))

'''
梯度上升法
'''
def gradient_ascent(dataMatInput,labelX):
    dataMat = np.matrix(dataMatInput)
    labelMat=np.mat(labelX).T
    m,n = np.shape(dataMatInput)
    alpha=0.001#初始化步长
    weights=np.ones((n,1))#初始化权重矩阵，暂时全部默认是1
    maxCycle=500#设置最大循环条件
    for i in range(maxCycle):
        h=sigmoid(dataMat*weights)
        error = labelMat-h #误差
        weights = weights+alpha*dataMat.T*error
    return weights


'''
随机梯度上升法(数据集分类效果好用此法)
'''
def gradient_random_ascent(dataMatInput,labelX):
    m,n=np.shape(dataMatInput)
    alpha=0.001
    # alpha=0.01
    weights=np.ones((n,1))
    for i in range(n):
        h=sigmoid(sum(dataMatInput[i]*weights))
        error=labelX[i]-h
        weights=weights+alpha*error*dataMatInput[i]
    return weights

'''
改进随即梯度上升法（数据集分类效果不好用此法）
iterNum是循环迭代次数
'''
def gradient_random_acent_improment(dataMatInput,labelX,iterNum=500):
    m,n=np.shape(dataMatInput)
    weights=np.ones((n))
    dataIndex=range(m)
    for i in range(iterNum):
        for j in range(m):
            alpha=4.0/(1.0+i+j)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatInput[randIndex]*weights))
            error=labelX[randIndex]-h
            weights = weights + alpha * error *dataMatInput[randIndex]
            del(dataMatInput[randIndex])
    return weights

'''
数据可视化
'''
def plotBestFit(weightsX):
    import matplotlib.pyplot as plot
    dataMat,labelMat=load_data()
    dataArray=np.array(dataMat)
    n=np.shape(dataArray)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArray[i,1]);ycord1.append(dataArray[i,2])
        else:
            xcord2.append(dataArray[i,1]);ycord2.append(dataArray[i,2])
    figure=plot.figure()
    ax=figure.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-1*weightsX[0]-weightsX[1]*x)/weightsX[2]
    ax.plot(x,y)
    plot.xlabel("X1");plot.ylabel("X2")
    plot.show()

