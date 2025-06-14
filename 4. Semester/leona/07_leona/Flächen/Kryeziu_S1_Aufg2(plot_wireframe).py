import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameter
c = 1  # Wellengeschwindigkeit
x = np.linspace(-10, 10, 100)
t = np.linspace(-10, 10, 100)
X, T = np.meshgrid(x, t)

# Funktionen
w = np.sin(X + c*T)
v = np.sin(X + c*T) + np.cos(2*X + 2*c*T)

# 3D-Plot
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Plot für w(x,t)
ax1.plot_wireframe(X, T, w, color='b')
ax1.set_title("w(x,t) = sin(x + ct)")
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.set_zlabel("w")

# Plot für v(x,t)
ax2.plot_wireframe(X, T, v, color='r')
ax2.set_title("v(x,t) = sin(x + ct) + cos(2x + 2ct)")
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax2.set_zlabel("v")

plt.show()
