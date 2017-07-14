from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris.data
data = data[:,0:2]
print(data.shape)
labels = iris.target_names

kmeans = KMeans(n_clusters=2,random_state=0).fit(data)
print(kmeans.set_params(n_clusters=3))
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)
# print(data)

plt.figure(figsize=(8, 5), dpi=80)
axes = plt.subplot(111)

type1_x,type1_y,labels_x,labels_y = [],[],[],[]
for list in data:
    type1_x.append(list[0])
    type1_y.append(list[1])
for list in kmeans.cluster_centers_:
    labels_x.append(list[0])
    labels_y.append(list[1])
type1 = axes.scatter(type1_x, type1_y,c='red' )
type2 = axes.scatter(labels_x, labels_y,c='green')
axes.legend((type1, type2), ('data', 'center'),loc=1)
# print(type(kmeans.labels_))
plt.show()

