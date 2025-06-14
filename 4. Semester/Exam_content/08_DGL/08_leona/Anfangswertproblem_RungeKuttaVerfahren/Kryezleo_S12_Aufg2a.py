import numpy as np
import matplotlib.pyplot as plt

# Klassische Runge-Kutta 4. Ordnung
def RK4(f, a, b, h, y0):
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, y

# Anfangswertproblem: y' = 1 - y/t, y(1) = 5
def f(t, y):
    return 1 - y / t

# Exakte Lösung
def y_exact(t):
    return t / 2 + 9 / (2 * t)

# Parameter
a = 1
b = 6
h = 0.01
y0 = 5

# Numerische Lösung mit klassischem RK4
x_vals, y_vals = RK4(f, a, b, h, y0)

# Exakte Lösung berechnen
y_true = y_exact(x_vals)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Numerische Lösung (RK4)", color="blue")
plt.plot(x_vals, y_true, label="Exakte Lösung", linestyle="dashed", color="green")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Vergleich: Numerische vs. Exakte Lösung")
plt.grid(True)
plt.legend()
plt.show()
