import numpy as np
import matplotlib.pyplot as plt

# Gegebene DGL: dy/dx = x^2 / y
def f(x, y):
    return x**2 / y

# Exakte Lösung: y(x) = sqrt((2/3)x^3 + 4)
def y_exact(x):
    return np.sqrt((2/3) * x**3 + 4)

# Euler-Verfahren
def euler(f, a, b, h, y0):
    n = int((b - a) / h)
    x = np.linspace(a, b, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

# Modifiziertes Euler-Verfahren (Heun)
def mod_euler(f, a, b, h, y0):
    n = int((b - a) / h)
    x = np.linspace(a, b, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i+1], y[i] + h * k1)
        y[i+1] = y[i] + h * (k1 + k2) / 2
    return x, y

# Mittelpunktverfahren
def midpoint(f, a, b, h, y0):
    n = int((b - a) / h)
    x = np.linspace(a, b, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h/2 * k1)
        y[i+1] = y[i] + h * k2
    return x, y

# Klassisches Runge-Kutta-Verfahren
def rk4(f, a, b, h, y0):
    n = int((b - a) / h)
    x = np.linspace(a, b, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x, y

# Parameter
a = 0
b = 10
h = 0.1
y0 = 2

# Lösungen berechnen
x_euler, y_euler = euler(f, a, b, h, y0)
x_mod, y_mod = mod_euler(f, a, b, h, y0)
x_mid, y_mid = midpoint(f, a, b, h, y0)
x_rk4, y_rk4 = rk4(f, a, b, h, y0)
y_true = y_exact(x_rk4)

# Plot der numerischen Lösungen
plt.figure(figsize=(10, 6))
plt.plot(x_euler, y_euler, label="Euler", linestyle='--')
plt.plot(x_mod, y_mod, label="Mod. Euler")
plt.plot(x_mid, y_mid, label="Mittelpunkt")
plt.plot(x_rk4, y_rk4, label="RK4")
plt.plot(x_rk4, y_true, label="Exakt", linestyle='dotted', color='black')
plt.title("Lösungsvergleich der Verfahren")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.show()

# Fehlervergleich (logarithmisch)
err_euler = np.abs(y_true - y_euler)
err_mod   = np.abs(y_true - y_mod)
err_mid   = np.abs(y_true - y_mid)
err_rk4   = np.abs(y_true - y_rk4)

plt.figure(figsize=(10, 6))
plt.semilogy(x_rk4, err_euler, label="Fehler Euler", linestyle='--')
plt.semilogy(x_rk4, err_mod, label="Fehler Mod. Euler")
plt.semilogy(x_rk4, err_mid, label="Fehler Mittelpunkt")
plt.semilogy(x_rk4, err_rk4, label="Fehler RK4")
plt.title("Globaler Fehlervergleich (log)")
plt.xlabel("x")
plt.ylabel("|y_exact - y_i|")
plt.legend()
plt.grid(True, which="both", linestyle='dotted')
plt.show()
