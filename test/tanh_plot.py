import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,1000)
x1 = np.linspace(-9,0,10)
y1 =[0,0,0,0,0,0,0,0,0,0]
#y = (np.e ** x - np.e ** -x) / (np.e ** x + np.e ** -x) #tanh
# y = 1/ (1 + np.e ** -x) #sigmoid
y = x
plt.plot(x,y,"k-",label='ReLU')
plt.plot(x1,y1,"k-",)

# plt.plot(x,1-y**2,'k--',label='derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.show()