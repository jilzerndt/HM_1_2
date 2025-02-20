import numpy as np

L = np.array([    [1, 0, 0],    [2, 1, 0],    [4, 2/(2**(-52)), 1]], dtype=np.float64)
R = np.array([    [1, 1, 1],    [0, 2**(-52), 3],    [0, 0, 4 - (6/(2**(-52)))]], dtype=np.float64)
b = np.array([1, 0, 0], dtype=np.float64)

y = np.linalg.solve(L, b)
print(y)
x1 = np.linalg.solve(R, y)
print(x1)