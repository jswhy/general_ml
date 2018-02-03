import pandas as pd
s = pd.Series(['Hello', 'World', "I'm", 'pandas'])
list = ['python', 24, 'program language']
dict = {'name':'python', 'age':24, 'type':'program language'}
dict_num = {'a':20, 'b':32, 'c':10, 'd':100}
# print(s)
# dic = {"name":['Lucky', "Happy"], 'age':[23,25], 'sex':['male','female']}
# df = pd.DataFrame(dic)
# print(s.index)
# s2 = pd.Series(dict, index=['name', 'age', 'type', "another"])
# print(s2)
# print(s2[s2>20])
s3 = pd.Series(dict_num)
# print(s3[20 < s3])
print(dict.get('name'))
print('name' in dict.keys())
