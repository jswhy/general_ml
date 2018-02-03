import pandas as pd
import numpy as np
A = np.ones((3, 4)) * 1
B = np.ones((3, 4)) * 2
C = np.ones((3, 4)) * 3
print(np.concatenate([A, B, C], axis=0), '\n')
A1 = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
B1 = pd.DataFrame(B, columns=['a', 'b', 'c', 'd'])
C1 = pd.DataFrame(C, columns=['a', 'b', 'c', 'd'])
print(pd.concat((A1, B1, C1), axis=1))