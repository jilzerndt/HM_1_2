import numpy as np
import matplotlib.pyplot as plt

def Gruppe11_S5_Aufg2(x, y, xx):
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n - 1, n - 1))
    rhs = np.zeros(n - 1)

    for i in range(1, n):
        A[i - 1, i - 1] = 2 * (h[i - 1] + h[i])
        if i > 1:
            A[i - 1, i - 2] = h[i - 1]
        if i < n - 1:
            A[i - 1, i] = h[i]
        rhs[i - 1] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    m = np.zeros(n + 1)
    if n > 1:
        m[1:n] = np.linalg.solve(A, rhs)

    yy = np.zeros_like(xx)
    for j, x_val in enumerate(xx):
        i = np.searchsorted(x, x_val) - 1
        i = np.clip(i, 0, n - 1)
        dx = x_val - x[i]
        hi = h[i]

        # Koeffizienten
        a = (m[i + 1] - m[i]) / (6 * hi)
        b = m[i] / 2
        c = (y[i + 1] - y[i]) / hi - (2 * hi * m[i] + hi * m[i + 1]) / 6
        d = y[i]

        yy[j] = a * dx**3 + b * dx**2 + c * dx + d


    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'ro', label='Stützpunkte')
    plt.plot(xx, yy, 'b-', label='Splineinterpolation')
    plt.xlabel('x')
    plt.ylabel('S(x)')
    plt.title('Natürliche kubische Splineinterpolation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return yy


# Daten von aufgabe 1
x = np.array([4, 6, 8, 10])
y = np.array([6, 3, 9, 0])
xx = np.linspace(4, 10, 200)

yy = Gruppe11_S5_Aufg2(x, y, xx)
