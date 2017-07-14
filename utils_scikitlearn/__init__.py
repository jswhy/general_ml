#coding=utf-8
import numpy as np
import urllib.request
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV

#初始化，从url的地址中读取测试数据
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
X = dataset[:,0:7]
y = dataset[:,8]

# 数据归一化，调用scikit-learn的normalization方法
normalized_X = preprocessing.normalize(X)
standardized_X = preprocessing.scale(X)

# 特征选择
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)# 计算特征值的信息熵


"""
逻辑回归
"""
model = LogisticRegression()
model.fit(X, y)
print(model)
expected = y
predicted = model.predict(X)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


"""
朴素贝叶斯
"""
model = GaussianNB()
model.fit(X, y)
print(model)
expected = y
predicted = model.predict(X)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

"""
KNN
"""
model = KNeighborsClassifier()
model.fit(X, y)
print(model)
expected = y
predicted = model.predict(X)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


"""
决策树
"""
model = DecisionTreeClassifier()
model.fit(X, y)
print(model)
expected = y
predicted = model.predict(X)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

"""
SVM
"""
model = SVC()
model.fit(X, y)
print(model)
expected = y
predicted = model.predict(X)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

"""
参数优化，岭回归
"""
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X, y)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.alpha)