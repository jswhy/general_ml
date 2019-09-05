import pandas as pd
import numpy as np
A = np.ones((3, 4)) * 1
B = np.ones((3, 4)) * 2
C = np.ones((3, 4)) * 3
# print(np.concatenate([A, B, C], axis=0), '\n')
A1 = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'], index=[0, 1, 2])
B1 = pd.DataFrame(B, columns=['b', 'c', 'd', 'e'], index=[0, 1, 2])
# C1 = pd.DataFrame(C, columns=['c', 'd', 'e', 'f'], index=[2, 3, 4])
print(pd.merge(A1, B1, on = ['b', 'c', 'd'], how='outer', suffixes=['left', 'right']))