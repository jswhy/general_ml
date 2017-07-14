from sklearn import linear_model
ll = linear_model.LinearRegression(fit_intercept=False).fit([[0],[1.1],[2.2]],[0,0.9,1.9])
print(ll.coef_)