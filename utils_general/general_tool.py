#coding=utf-8
import numpy as np
#通用工具类函数库

'''
初始化函数，用来读取文件，并将文件拆分为数据集和标签集
输入文件为以逗号分割的多行txt文件，其值分别为特征值1和特征值2，第三列为标签
'''
def load_data(x0=1.0):
    dataMat,labelMat = [],[];
    file = open('readTxt.txt')
    for line in file.readlines():
        lineArr = line.strip().split()
        # lineArr = line.strip().split(',')
        dataMat.append([x0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

'''
归一化特征值函数(0,1标准化)
value := (value-min)/(max-min),其中min和max分别是数据集中的最大值和最小值
'''
def autoNorm(dataSet):
    minValue=dataSet.min(0)
    maxValue=dataSet.max(0)
    ranges=maxValue-minValue
    normDataSet=np.zeros(dataSet.shape)
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minValue,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minValue

# #归一化特征值函数（均值归一化）
# # x:= (x-avg)/sigma
# def meanNorm(dataSet):
#     (m,n)=np.array(dataSet).shape
#     dataSet=dataSet.tranpose()
#     resultSet=np.ones((m,n))
#     for i in range(n):
#         sigma=np.std(dataSet,axis=i)
#         print(sigma)
# a=np.array([[1,2,3],[4,5,6],[7,8,9]])
# meanNorm(a)