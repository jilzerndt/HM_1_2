import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PARAMETER – HIER ANPASSEN FÜR DIE AUFGABE
# ============================================

stufen = 3           # <== HIER einstellen: 2, 3, 4 oder 5 Stufen(Ursprünglich für Stufen 3 gemacht)
t_start = 2          # <== Anfangszeit t₀
t_end = 5            # <== Endzeitpunkt
y0 = 1               # <== Anfangswert y(t₀)
h_values = [1, 0.1, 0.01, 0.001]  # <== Schrittweiten für Konvergenzuntersuchung

# ============================================
# Rechte Seite der DGL: y' = f(t, y)
# ============================================
def f(t, y):
    return t / y  # <== HIER f(t, y) ändern, wenn DGL anders

# ============================================
# Exakte Lösung – falls bekannt
# ============================================
def y_exact(t):
    return np.sqrt(t**2 - 3)  # <== HIER anpassen bei anderer exakter Lösung

# ============================================
# Runge-Kutta-Verfahren mit 2–5 Stufen
# ============================================
def runge_kutta(f, t0, y0, t_end, h, stufen):
    t_values = [t0]
    y_values = [y0]
    t = t0
    y = y0
    while t < t_end - 1e-10:
        if stufen == 2:
            # Heun-Verfahren
            k1 = f(t, y)
            k2 = f(t + 0.5*h, y + 0.5*h*k1)
            y = y + h * k2
        elif stufen == 3:
            # 3-stufiges Verfahren (aus Aufgabe)
            k1 = f(t, y)
            k2 = f(t + h/3, y + h/3 * k1)
            k3 = f(t + 2*h/3, y + 2*h/3 * k2)
            y = y + h * (1/4 * k1 + 3/4 * k3)
        elif stufen == 4:
            # Klassisches RK4-Verfahren
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)
        elif stufen == 5:
            # Klassisches RK5-Verfahren (Butcher's RK5)
            k1 = f(t, y)
            k2 = f(t + h/4, y + h/4 * k1)
            k3 = f(t + h/4, y + h/8 * k1 + h/8 * k2)
            k4 = f(t + h/2, y - h/2 * k2 + h * k3)
            k5 = f(t + 3*h/4, y + 3*h/16 * k1 + 9*h/16 * k4)
            k6 = f(t + h, y - 3*h/7 * k1 + 2*h/7 * k2 + 12*h/7 * k3 - 12*h/7 * k4 + 8*h/7 * k5)
            y = y + h * (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6) / 90
        else:
            raise ValueError("Nur 2–5 Stufen werden unterstützt.")
        t = t + h
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

# ============================================
# Plot der numerischen & exakten Lösung
# ============================================
h_plot = 0.1  # <== Schrittweite für Plot (nur für Teil b)
t_vals, y_vals = runge_kutta(f, t_start, y0, t_end, h_plot, stufen)

plt.plot(t_vals, y_vals, label=f'Numerisch (h = {h_plot})')
t_exact = np.linspace(t_start, t_end, 300)
plt.plot(t_exact, y_exact(t_exact), '--', label='Exakte Lösung')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f'Numerische Lösung (RK mit {stufen} Stufen)')
plt.legend()
plt.grid(True)
plt.show()

# ============================================
# Fehleranalyse & Konvergenzordnung (Teil c)
# ============================================

errors = []

for h in h_values:
    _, y_vals = runge_kutta(f, t_start, y0, t_end, h, stufen)
    y_num = y_vals[-1]
    y_ex = y_exact(t_end)
    err = abs(y_num - y_ex)
    errors.append(err)

# Berechne p (nur aus den letzten zwei Werten)
h1, h2 = h_values[-2], h_values[-1]
e1, e2 = errors[-2], errors[-1]
p = (np.log(e1) - np.log(e2)) / (np.log(h1) - np.log(h2))

# Fehlerplot
plt.loglog(h_values, errors, 'o-', base=10)
plt.xlabel('h (Schrittweite)')
plt.ylabel(f'Fehler bei t = {t_end}')
plt.title(f'Konvergenzordnung p ≈ {p:.2f} (RK mit {stufen} Stufen)')
plt.grid(True, which="both", ls="--")
plt.show()

# Konsole
print(f"Verwendetes Verfahren: Runge-Kutta mit {stufen} Stufen")
print(f"Konvergenzordnung p ≈ {p:.5f}")
