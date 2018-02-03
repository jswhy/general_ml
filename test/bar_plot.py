import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(1, 6, 6)
x1 = np.linspace(1,3,3)
x2=np.linspace(1, 4, 4)
#
# y = [55767, 17570, 9545, 6778, 6196, 4320]
# y1 = [222, 57, 16]
# y2 = [0.859,0.8218,0.84]
# y3 = [0.7978, 0.7804, 0.7886]

y21 = [19.8, 33.5, 43, 55]
# plt.bar(x, y, color = 'black', edgecolor = 'black', width=0.3)
# plt.bar(x1, y2, color = 'black', edgecolor = 'black', width=0.2)
# plt.bar(x1+0.2, y3, color = 'white', edgecolor = 'black', width=0.2)
# for x, y in zip(x, y):
#     plt.text(x, y + 0.1, '%s' % y, ha = 'center', va = 'bottom')

# plt.xticks([1, 2, 3, 4, 5, 6], [r'$TensorFlow$', r'$Caffe$', r"$MXNeT$",r'$Torch$',r'$Theano$',r'$Caffe2$'])
plt.bar(x2, y21, color = 'c', edgecolor = 'white', width=0.3)
# plt.xticks([1.1, 2.1, 3.1],[r'$Precision$', r'$Recall$', r'$F1\ Score$'])
plt.ylabel("percent")
plt.xticks([1, 2, 3,4],[r'$RNNLM$', r'$NNLM$', r'$CBOW$', r'$Skip-gram$'])

plt.show()
