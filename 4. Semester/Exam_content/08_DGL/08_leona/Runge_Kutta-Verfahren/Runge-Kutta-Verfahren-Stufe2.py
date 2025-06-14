import numpy as np
import matplotlib.pyplot as plt

# Anfangsbedingungen
x0 = 0
x_end = 2
y0 = 1
h = 0.1

# Anzahl Schritte
n = int((x_end - x0) / h) + 1

# Arrays zur Speicherung
x_vals = np.linspace(x0, x_end, n)
y_vals = np.zeros(n)
y_vals[0] = y0

# Definition der DGL
def f(x, y):
    return x**2 - y

# Runge-Kutta-Verfahren 2. Ordnung mit gegebenem Butcher-Schema
for i in range(n - 1):
    k1 = f(x_vals[i], y_vals[i])
    k2 = f(x_vals[i] + 2/3 * h, y_vals[i] + 2/3 * h * k1)
    y_vals[i+1] = y_vals[i] + h * (1/4 * k1 + 3/4 * k2)

# Richtungspfeile vorbereiten (Richtungsfeld)
X, Y = np.meshgrid(np.arange(0, 2.25, 0.25), np.arange(0, 2.25, 0.25))
U = 1
V = f(X, Y)
N = np.sqrt(U**2 + V**2)
U, V = U/N, V/N  # normierte Pfeile

# Plot
plt.figure(figsize=(8, 5))
plt.quiver(X, Y, U, V, angles="xy")
plt.plot(x_vals, y_vals, 'r-', label="numerische Lösung (RK2)")
plt.title("Numerische Lösung mit Runge-Kutta 2. Ordnung + Richtungsfeld")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([0, 2])
plt.ylim([0, 2])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
