import numpy as np
import matplotlib.pyplot as plt

# === Parameter (für andere Aufgabe anpassen) ===
m = 97000           # Masse in kg (hier konstant)
v0 = 100            # Anfangsgeschwindigkeit in m/s
x0 = 0              # Anfangsposition in m
t_start = 0         # Startzeitpunkt
t_end = 20          # Endzeitpunkt (anpassen, wenn nötig)
dt = 0.1            # Schrittweite (∆t)

# === Differentialgleichung definieren ===
# Die rechte Seite der 2. Ordnung DGL: m * a = -5*v^2 - 0.1*x - 570000
# wird umgeformt zu a = f(x, v)
def acceleration(x, v):
    return (-5 * v**2 - 0.1 * x - 570000) / m
    # <- Diese Funktion anpassen, wenn du eine andere DGL hast!

# === Zeitschritte vorbereiten ===
t_vals = np.arange(t_start, t_end + dt, dt)
x_vals = np.zeros_like(t_vals)
v_vals = np.zeros_like(t_vals)

# === Anfangswerte setzen ===
x_vals[0] = x0
v_vals[0] = v0

# === Mittelpunktverfahren zur Lösung der gekoppelten DGLs ===
for i in range(len(t_vals) - 1):
    t = t_vals[i]
    x = x_vals[i]
    v = v_vals[i]

    # Schritt 1: Halbzeitschritt berechnen
    a1 = acceleration(x, v)
    v_half = v + 0.5 * dt * a1
    x_half = x + 0.5 * dt * v

    # Schritt 2: Endwert mit Mittelwert berechnen
    a2 = acceleration(x_half, v_half)
    v_vals[i + 1] = v + dt * a2
    x_vals[i + 1] = x + dt * v_half

# === Plot: Position und Geschwindigkeit über der Zeit ===
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x_vals, label='x(t) [Position]')
plt.plot(t_vals, v_vals, label='v(t) [Geschwindigkeit]')
plt.title("Mittelpunktverfahren: Bremsweg einer Boeing 737-200")
plt.xlabel("Zeit t [s]")
plt.ylabel("x(t), v(t)")
plt.grid(True)
plt.legend()
plt.show()

# === Aufgabe c: Stillstand erkennen (v ≈ 0) und Werte auslesen ===
for i in range(1, len(v_vals)):
    if v_vals[i] <= 0:
        print(f"Stillstand bei t = {t_vals[i]:.2f} s, Bremsweg = {x_vals[i]:.2f} m")
        break

# === Kommentare zur Anpassung für andere Aufgaben ===
# - acceleration(x, v): Hier die neue DGL einsetzen
# - m, v0, x0: Anfangswerte neu setzen
# - t_start, t_end, dt: Zeitintervall und Auflösung anpassen
