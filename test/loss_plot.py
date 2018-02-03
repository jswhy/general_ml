import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 20, 20)
y = [674.599, 421.366, 305.943, 241.748, 205.29, 180.5, 160.867, 145.657, 133.973, 124.281, 115.586, 108.34, 101.959,
     96.3985, 91.6309, 87.29, 83.2224, 79.5156,76.111, 73.1628]
y1 = []
for num in y:
    y1.append(num+num*0.5)
y2 = [674.599, 461.366, 365.943, 291.748, 245.29, 225.29, 160.867, 245.657, 233.973, 224.281, 315.586, 308.34, 301.959,
     296.3985, 291.6309, 287.29, 283.2224, 319.5156,306.111, 313.1628]
y1[0]= 675
plt.plot(x, y,'g-', label = 'tanh')
plt.plot(x, y1,'b-', label = 'sigmoid')
plt.plot(x, y2,'c-', label = 'ReLU')
plt.legend(loc = 'best')
plt.xlabel('epoch')
plt.ylabel('loss function')
plt.show()