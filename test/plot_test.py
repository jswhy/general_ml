import matplotlib.pyplot as plt
import numpy as np
from numpy import random

x = np.linspace(1, 20, 20)
print(x)
PI = np.pi /4
# y = (1 - 1 / (1 + np.exp(x * -1 / 6))) * 1000 + 150 + 2 * x
# line = plt.plot(x, y, 'k-', label = 'ReLU')
# y2 = (1 - 1 / (1 + np.exp(x * -1 / 10))) * 1000 + 88
# line2 = plt.plot(x, y2, 'b-',label = 'Sigmoid')
# y3 = (1 - 1 / (1 + np.exp(x * -1 / 7))) * 1000 + 66
# line3 = plt.plot(x, y3, 'r-',label = 'tanh')
# y2 = [674.599, 421.366, 305.943, 241.748, 205.29, 180.5, 160.867, 145.657, 133.973, 124.281, 115.586, 108.34, 101.959,
#          96.3985, 91.6309, 87.29, 83.2224, 79.5156,76.111, 73.1628]
# y1 = [300, 145, 132, 179, 133,113,87,112,94,81,74,86,74,66,61,58,40,33,41,30]
# y1 = [974.599, 921.366, 835.943, 791.748, 725.29, 680.5, 620.867, 575.657, 533.973, 474.281, 415.586, 368.34, 301.959,
#          256.3985, 191.6309, 167.29, 102.2224, 72.5156, 81.111, 75.1628]
# y2 = [974.599, 821.366, 764.943, 641.748, 495.29, 410.5, 360.867, 298.657, 299.973, 266.281, 285.586, 274.34, 240.159,
#          196.3985, 191.6309, 187.29, 183.2224, 179.5156,176.111, 173.1628]
y = np.sin(x * PI) + 3
y3, y4 = [], []
for i in range(20):
    y3.append(3)
    if 0 == x[i]%2:
        y4.append(y[i] + random.random())
    elif 0 != x[i]%2:
        y4.append(y[i] - random.random())
# y3, y1, y4, y5 = [], [], [],[]
# for i in range(20):
#     if i<10:
#         y1.append(y2[i] - random.random() * y2[i] + 10)
#         y3.append(y2[i] - random.random() * y2[i] + 20)
#         y5.append(y2[i] + random.random() * y2[i] - 10)
#         y4.append(y2[i] + random.random() * y2[i] - 20)
#     else:
#         y1.append(y2[i] + random.random() * y2[i] + 10)
#         y3.append(y2[i] + random.random() * y2[i] + 20)
#         y4.append(y2[i] + random.random() * y2[i] - 10)* 0.3)
#         y5.append(y2[i] + random.random() * y2[i] - 20)


plt.plot(x, y,'r-', label = 'overfitting')
plt.plot(x, y3,'b-', label = 'fitting')
plt.scatter(x, y4)
# plt.plot(x, y2,'g-', label = '100')
# plt.plot(x, y3,'k-', label = '50')
# plt.plot(x, y4,'b-', label = '200')
# plt.plot(x, y5,'y-', label = '150')
plt.legend(loc = 'best')
plt.ylim(0, 7)
plt.show()

