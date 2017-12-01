import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 5, 6)
print(x)
y = [55767, 17570, 9545, 6778, 6196, 4320]
plt.bar(x, y, color = 'c', edgecolor = 'white', width=0.4)
for x, y in zip(x, y):
    plt.text(x + 0.05, y + 0.1, '%d' % y, ha = 'center', va = 'bottom')
new_ticks = ["TensorFlow", "Caffe", "MXNeT", "Torch", "Theano", "Caffe2"]
plt.xticks([0, 1, 2, 3, 4, 5],[r'$TensorFlow$', r'$Caffe$', r"$MXNeT$", r"$Torch$", r"$Theano$", r"$Caffe2$"] )
plt.show()
