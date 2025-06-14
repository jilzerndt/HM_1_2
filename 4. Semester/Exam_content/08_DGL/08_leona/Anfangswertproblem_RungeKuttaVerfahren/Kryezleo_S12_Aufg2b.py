import numpy as np

def RK4_custom(f, a, b, h, y0):
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    # Butcher-Koeffizienten
    c2, c3, c4 = 0.25, 0.5, 0.75
    a21 = 0.25
    a31, a32 = 0.0, 0.5
    a41, a42, a43 = 0.0, 0.0, 0.75
    b1, b2, b3, b4 = 0.1, 0.2, 0.3, 0.4

    for i in range(n):
        xi = x[i]
        yi = y[i]

        k1 = h * f(xi, yi)
        k2 = h * f(xi + c2 * h, yi + a21 * k1)
        k3 = h * f(xi + c3 * h, yi + a31 * k1 + a32 * k2)
        k4 = h * f(xi + c4 * h, yi + a41 * k1 + a42 * k2 + a43 * k3)

        y[i+1] = yi + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4

    return x, y
