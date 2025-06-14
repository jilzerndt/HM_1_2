import numpy as np
import matplotlib.pyplot as plt

def Gruppe11_S11_Aufg1(f, xmin, xmax, ymin, ymax, hx, hy):
    x = np.arange(xmin, xmax + hx, hx)
    y = np.arange(ymin, ymax + hy, hy)
    X, Y = np.meshgrid(x, y)
    
    U = np.ones_like(X)  # x-Komponente des Richtungsfelds
    V = f(X, Y)          # y-Komponente (Steigung)

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, angles='xy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Richtungsfeld der DGL')
    plt.grid(True)
    plt.show()

def f(x, y):
    return y - x

Gruppe11_S11_Aufg1(f, xmin=0, xmax=2, ymin=0, ymax=2, hx=0.2, hy=0.2)
