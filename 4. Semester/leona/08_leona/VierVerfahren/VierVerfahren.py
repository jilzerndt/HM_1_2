import numpy as np
import matplotlib.pyplot as plt

# ============================================
# HIER BEGINNT DER BEREICH, DEN DU BEI ANDERER AUFGABE ÄNDERN MUSST:
# ============================================

h = 0.1                      # Schrittweite (z.B. bei anderer Genauigkeit ändern)
x = np.arange(0, 10 + h, h)  # Intervall [x0, xmax] (bei anderer Aufgabe anpassen)
y0 = 2                      # Anfangsbedingung y(x0) = y0 (aus Aufgabenstellung übernehmen)

# Die rechte Seite der DGL dy/dx = f(x, y)
def f(x, y):
    return x**2 / y         # HIER die DGL anpassen!

# Exakte Lösung der DGL (wenn vorhanden, sonst entfernen)
def y_exact(x):
    return np.sqrt((2/3) * x**3 + 4)  # HIER exakte Lösung anpassen!

# ============================================
# HIER ENDET DER BEREICH, DEN DU BEI ANDERER AUFGABE ÄNDERN MUSST
# ============================================


# ---- Numerische Verfahren (bleiben gleich) ----

def euler(f, x, y0):
    y = [y0]
    for i in range(len(x) - 1):
        y.append(y[-1] + h * f(x[i], y[-1]))
    return np.array(y)

def midpoint(f, x, y0):
    y = [y0]
    for i in range(len(x) - 1):
        k1 = f(x[i], y[-1])
        k2 = f(x[i] + h/2, y[-1] + h/2 * k1)
        y.append(y[-1] + h * k2)
    return np.array(y)

def mod_euler(f, x, y0):
    y = [y0]
    for i in range(len(x) - 1):
        k1 = f(x[i], y[-1])
        k2 = f(x[i+1], y[-1] + h * k1)
        y.append(y[-1] + h/2 * (k1 + k2))
    return np.array(y)

def runge_kutta(f, x, y0):
    y = [y0]
    for i in range(len(x) - 1):
        k1 = f(x[i], y[-1])
        k2 = f(x[i] + h/2, y[-1] + h/2 * k1)
        k3 = f(x[i] + h/2, y[-1] + h/2 * k2)
        k4 = f(x[i] + h, y[-1] + h * k3)
        y.append(y[-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4))
    return np.array(y)

# ---- Lösungen berechnen ----
y_euler = euler(f, x, y0)
y_mid = midpoint(f, x, y0)
y_mod = mod_euler(f, x, y0)
y_rk = runge_kutta(f, x, y0)
y_ref = y_exact(x)

# ---- Fehler berechnen (wenn exakte Lösung vorhanden ist) ----
error_euler = np.abs(y_ref - y_euler)
error_mid = np.abs(y_ref - y_mid)
error_mod = np.abs(y_ref - y_mod)
error_rk = np.abs(y_ref - y_rk)

# ---- Plot der Lösungen ----
plt.figure()
plt.plot(x, y_ref, 'k', label='exakt')  # Exakte Lösung
plt.plot(x, y_euler, '--', label='Euler')
plt.plot(x, y_mid, '--', label='Mittelpunkt')
plt.plot(x, y_mod, '--', label='mod. Euler')
plt.plot(x, y_rk, '--', label='Runge-Kutta')
plt.title("Lösungen der DGL")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# ---- Plot der Fehler (log y-Achse) ----
plt.figure()
plt.semilogy(x, error_euler, label='Euler')
plt.semilogy(x, error_mid, label='Mittelpunkt')
plt.semilogy(x, error_mod, label='mod. Euler')
plt.semilogy(x, error_rk, label='Runge-Kutta')
plt.title("Globaler Fehler |y(x) - yᵢ|")
plt.xlabel("x")
plt.ylabel("Fehler (log)")
plt.legend()
plt.grid()

plt.show()

# ---- Theoretischer Kommentar zu Teil b ----
print("\nOptional Teil b:")
print("Um den Fehler von Runge-Kutta mit Euler zu erreichen,")
print("müsste man die Schrittweite bei Euler ca. um den Faktor 10–100 verkleinern.")
print("Denn: Euler hat O(h), Runge-Kutta hat O(h^4) Fehlerordnung.")
