import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 2, 3)
print(x)

y = [142, 54, 6]

plt.bar(x, y, color = 'c', edgecolor = 'white', width=0.3)


# for x, y in zip(x, y):
#     plt.text(x + 0.05, y + 0.1, '%s' % y, ha = 'center', va = 'bottom')

plt.xticks([0, 1, 2,], [r'$TXT$', r'$MySQL$', r"$MongoDB$"])

plt.ylabel('time\ [s]')
plt.show()
