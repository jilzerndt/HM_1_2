import numpy as np
import matplotlib.pyplot as plt

# === Anfangswerte und Parameter ===
y0 = 0
t0 = 0
t_end = 6
h = 0.2

# === f(t, y): rechte Seite der DGL ===
def f(t, y):
    return 0.1 * y + np.sin(2 * t)
    # <- bei anderer Aufgabe hier die DGL anpassen

# === Euler-Verfahren ===
def euler(f, y0, t0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    for i in range(len(t_values) - 1):
        y_values[i + 1] = y_values[i] + h * f(t_values[i], y_values[i])
    return t_values, y_values

# === Klassisches Runge-Kutta (RK4) ===
def rk4(f, y0, t0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    for i in range(len(t_values) - 1):
        t = t_values[i]
        y = y_values[i]
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        y_values[i + 1] = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return t_values, y_values

# === a) Beide Näherungen berechnen ===
t_euler, y_euler = euler(f, y0, t0, t_end, h)
t_rk4, y_rk4 = rk4(f, y0, t0, t_end, h)

# === Plot beider Näherungen ===
plt.figure(figsize=(10, 5))
plt.plot(t_euler, y_euler, label="Euler", linestyle='--', marker='o')
plt.plot(t_rk4, y_rk4, label="Runge-Kutta 4", linestyle='-', marker='x')
plt.title("Numerische Lösungen mit h = 0.2")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend()
plt.show()

# === b) Fehlerplot (halblogarithmisch) ===
error = np.abs(y_rk4 - y_euler)

plt.figure(figsize=(10, 4))
plt.semilogy(t_euler, error, label="|RK4 - Euler|", color="red")
plt.title("Fehler zwischen Euler und RK4 (halb-log)")
plt.xlabel("t")
plt.ylabel("Fehlerbetrag")
plt.grid(True, which="both")
plt.legend()
plt.show()
