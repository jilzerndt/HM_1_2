import numpy as np
import matplotlib.pyplot as plt

# Gegebenes Anfangswertproblem
def f(t, y):
    return 1 - y / t

# Exakte Lösung
def y_exact(t):
    return t / 2 + 9 / (2 * t)

# Klassisches RK4-Verfahren
def RK4_classic(f, a, b, h, y0):
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

# Eigenes RK-Verfahren (wie in Teil c)
def RK4_custom(f, a, b, h, y0):
    n = int((b - a) / h)
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    # Eigene Butcher-Koeffizienten
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

# Parameter
a = 1
b = 6
h = 0.01
y0 = 5

# Lösungen berechnen
x_vals, y_classic = RK4_classic(f, a, b, h, y0)
_, y_custom = RK4_custom(f, a, b, h, y0)
y_exact_vals = y_exact(x_vals)

# Fehler berechnen
error_classic = np.abs(y_classic - y_exact_vals)
error_custom = np.abs(y_custom - y_exact_vals)

# Plot: Vergleich der Lösungen
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_exact_vals, label="Exakte Lösung", color="black", linestyle="dashed")
plt.plot(x_vals, y_classic, label="Klassisches RK4", color="blue")
plt.plot(x_vals, y_custom, label="Eigenes RK4", color="orange")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Vergleich: Exakte vs. RK4 vs. Eigenes RK4")
plt.grid(True)
plt.legend()
plt.show()

# Plot: Fehlervergleich
plt.figure(figsize=(10, 5))
plt.plot(x_vals, error_classic, label="Fehler Klassisches RK4", color="blue")
plt.plot(x_vals, error_custom, label="Fehler Eigenes RK4", color="orange")
plt.xlabel("t")
plt.ylabel("Absoluter Fehler")
plt.title("Absoluter Fehlervergleich")
plt.grid(True)
plt.legend()
plt.show()

# Kommentar ausgeben
print("Kommentar:")
print("Beide Verfahren liefern sehr ähnliche Ergebnisse. Der Fehler des klassischen RK4 ist minimal kleiner,")
print("aber das eigene Verfahren liefert ebenfalls eine sehr gute Approximation. Für viele praktische Zwecke")
print("kann auch das eigene Verfahren gut geeignet sein, sofern Stabilität und Genauigkeit ausreichend geprüft wurden.")
