import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Gegebene Zeitpunkte und Positionen
t = np.array([0, 0.5, 2, 3])
x = np.array([1, 2, 2.5, 0])

# Erzeuge natürlichen kubischen Spline
spline = CubicSpline(t, x, bc_type='natural')

# a) Koeffizienten berechnen
# scipy gibt sie in der Form: s(t) = c[0]*(t - t_k)^3 + c[1]*(t - t_k)^2 + c[2]*(t - t_k) + c[3]
coefficients = spline.c.T  # jede Zeile entspricht einem Intervall
t_intervals = t[:-1]       # zugehörige t_k-Werte, in Aufgabestellung herauslesen

# In DataFrame umwandeln für Übersicht
df = pd.DataFrame(coefficients, columns=["d_k", "c_k", "b_k", "a_k"])
df["t_k"] = t_intervals

# Position bei t = 1
t_eval = 1
position_at_1 = spline(t_eval)

# b) Geschwindigkeit und Beschleunigung bei t = 1
velocity_at_1 = spline(t_eval, 1)       # 1. Ableitung
acceleration_at_1 = spline(t_eval, 2)   # 2. Ableitung

# Geschwindigkeit über dichtes t für Plot
t_dense = np.linspace(t[0], t[-1], 200)
velocity_curve = spline(t_dense, 1)

# Ergebnisse ausgeben
print("Spline-Koeffizienten (je Intervall [t_k, t_{k+1}]):")
print(df.round(4))
print(f"\nPosition bei t = 1: {position_at_1:.4f}")
print(f"Geschwindigkeit bei t = 1: {velocity_at_1:.4f}")
print(f"Beschleunigung bei t = 1: {acceleration_at_1:.4f}")

# Plot der Geschwindigkeit
plt.figure(figsize=(8, 4))
plt.plot(t_dense, velocity_curve, label="Geschwindigkeit $\dot{x}(t)$")
plt.axvline(x=1, color='r', linestyle='--', label="t = 1")
plt.title("Verlauf der Geschwindigkeit $\dot{x}(t)$")
plt.xlabel("t")
plt.ylabel("$\dot{x}(t)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
