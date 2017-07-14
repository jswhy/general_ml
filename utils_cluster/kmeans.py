#coding=utf-8
import numpy as np


'''
读入csv文件并初始化为数据集
'''
def load_data(file="filename.csv"):
    dataMat=[]
    file=open(file)
    for line in file.readlines():
        curLine = line.strip().split(',')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat


'''
计算两个向量的欧氏距离
'''
def distEclud(vecA,vecB):
    return np.sqrt(sum(vecA-vecB,2))


'''
随机求质心函数
为给定数据集构建一个包含k个随机质心的集合
'''
def randCent(dataSet,k=2):
    n=np.shape(dataSet)[1]
    centroids=np.matrix(np.zeros(n,k))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*np.random.rand(k,1)
    return centroids


'''
kmeans主算法
'''
def kmeans (dataSet,k=2,distMess=distEclud,createCent=randCent):
    m=np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids=createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;
            minIndex = -1;
            for j in range(k):
                distJI=distMess(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    # 寻找最近的质心
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            # 更新质心的位置
            ptsInClust = dataSet[np.nonzero()]
            centroids[cent,:]=np.mean(ptsInClust,axis=0)
    return centroids,clusterAssment

