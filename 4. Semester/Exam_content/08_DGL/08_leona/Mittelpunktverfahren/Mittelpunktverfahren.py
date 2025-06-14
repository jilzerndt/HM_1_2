import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PARAMETER – HIER ANPASSEN!
# ===============================
m = 97000       # Masse des Objekts in kg
t0 = 0          # Startzeit in Sekunden
t_end = 20      # Endzeit der Simulation in Sekunden
dt = 0.1        # Zeitschrittweite in Sekunden

x0 = 0.0        # Anfangsposition in m
v0 = 100.0      # Anfangsgeschwindigkeit in m/s

# ===============================
# Wenn du eine andere Aufgabe lösen willst:
# - m anpassen (neue Masse)
# - x0, v0 anpassen (neue Anfangswerte)
# - t_end & dt ggf. ändern (längere Zeit oder feinere Auflösung)
# - Die Beschleunigungsfunktion a(x, v) unten anpassen,
#   falls sich die DGL oder Bremskraft ändert.
# ===============================

n_steps = int((t_end - t0) / dt)

# Arrays zur Speicherung
t = np.linspace(t0, t_end, n_steps + 1)
x = np.zeros(n_steps + 1)
v = np.zeros(n_steps + 1)

# Initialbedingungen
x[0] = x0
v[0] = v0

# ===============================
# BESCHLEUNIGUNGSFUNKTION – HIER ANPASSEN BEI ANDERER DGL!
# ===============================
def a(x, v):
    return ( -5 * v**2 - 0.1 * x - 570000 ) / m
# Diese Funktion a(x, v) definiert die rechte Seite der DGL
# Wenn die DGL anders aussieht, musst du diese Funktion ändern.
# ===============================

# Mittelpunktverfahren
for i in range(n_steps):
    x_mid = x[i] + 0.5 * dt * v[i]
    v_mid = v[i] + 0.5 * dt * a(x[i], v[i])

    x[i+1] = x[i] + dt * v_mid
    v[i+1] = v[i] + dt * a(x_mid, v_mid)

# Plotten
plt.figure(figsize=(10, 5))
plt.plot(t, x, label='x(t) Ort [m]')
plt.plot(t, v, label='v(t) Geschwindigkeit [m/s]')
plt.axhline(0, color='gray', linewidth=0.5)
plt.title("Landung einer Boeing 737-200 – Mittelpunktverfahren")
plt.xlabel("Zeit t [s]")
plt.ylabel("x(t), v(t)")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# AUSWERTUNG – ZEITPUNKT DES STILLSTANDS
# ===============================
for i in range(len(v)):
    if v[i] <= 0:
        print(f"Stillstand nach ca. {t[i]:.2f} s, Bremsweg ca. {x[i]:.2f} m")
        break


