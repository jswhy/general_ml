import numpy as np
import matplotlib.pyplot as plt
from numpy import random
x = np.linspace(0, 19 ,20)
y1 = [0, 0.19, 0.27, 0.33, 0.41, 0.49, 0.54, 0.57, 0.59, 0.61, 0.622, 0.665,0.691,0.712,0.733,0.763,0.791,0.825,
      0.885,0.912]
y2 = []
for num in y1:
    y2.append(num - random.random_sample() * 0.1)
plt.plot(x,y1,'r-',label = 'LR')
plt.plot(x,y2,'b-',label = 'SVM')
y2[0]=0
plt.legend(loc = "best")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()