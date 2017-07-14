# coding=utf-8
import math
import numpy as np

'''
非线性激励函数：sigmoid
'''
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
非线性激励函数：线性修正单元Rectified linear unit
'''
def relu(input):
    if type(input)==np.array or type(input)==np.matrix:
        if type(input)==np.array:
            np.mat(input)
        for i in input:
            print(i)
            if i>=0:
                continue
            else:
                i=0
            return input
    else:
        if input>=0:
            return input
        else:
            return 0


'''
非线性激励函数：正切函数tanh
'''
def tanh(input):
    return (1-math.exp(-2*input))/(1+math.exp(-2*input))


a=np.array([[1],[4],[-1]])
a=np.mat(a)
print (relu(a))