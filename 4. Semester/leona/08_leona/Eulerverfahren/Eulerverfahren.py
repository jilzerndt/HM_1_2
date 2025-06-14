import numpy as np
import matplotlib.pyplot as plt

# Definition der Differentialgleichung: y' = f(t, y)
def f(t, y):
    return 0.1 * y + np.sin(2 * t)

# Eulerverfahren
def euler(f, a, b, h, y0):
    n = int((b - a) / h)
    t = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y

# Parameter
a = 0       # Startzeit
b = 6       # Endzeit
h = 0.2     # Schrittweite
y0 = 0      # Anfangswert

# Numerische Lösung mit dem Eulerverfahren
t_vals, y_vals = euler(f, a, b, h, y0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_vals, y_vals, label="Numerische Lösung (Euler)", color="blue")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Numerische Lösung mit dem Eulerverfahren")
plt.grid(True)
plt.legend()
plt.show()