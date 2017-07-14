from sklearn import linear_model
import numpy as np
import random
ll = linear_model.LinearRegression()
list_result,result = [],[]
for i in range(100):
    list_result.append(i)
    result.append(2*i+random.uniform(-1,1)*0.1)
list_result = np.array(list_result).reshape(100,1)

llrg = linear_model.Ridge(alpha=.5)
llrg.fit(list_result,result)
print(llrg.coef_)
lasso = linear_model.Lasso(alpha=.6).fit(list_result,result)
print(lasso.coef_)

reg = linear_model.LinearRegression().fit(list_result,result)
print(reg.coef_)