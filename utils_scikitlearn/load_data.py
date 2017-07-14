from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import time
#读取鸢尾花数据集
dataResult=[]
iris = datasets.load_iris()
data = iris.data
data=list(data)
for i in range(100000):
    for j in  data:
        dataResult.append(j)
dataResult=np.array(dataResult)
print(dataResult.shape)

start = time.clock()
kmeans=KMeans(n_clusters=2,random_state=0).fit(dataResult)
end = time.clock()
print('finish all in %s' % str(end - start))
print(kmeans.inertia_)

