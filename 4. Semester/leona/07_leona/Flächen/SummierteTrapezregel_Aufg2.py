import numpy as np
import matplotlib.pyplot as plt

# ================================
# HIER BEGINNT DER BEREICH, DEN DU ANPASSEN MUSST, WENN DU EINE ANDERE AUFGABE HAST:
# ================================

g = 9.81  # m/s² - Erdanziehung, bleibt i.d.R. gleich
v_rel = 2600  # m/s - Ausströmgeschwindigkeit des Treibstoffs (aus der Aufgabenstellung übernehmen!)
m_A = 300000  # kg - Startmasse der Rakete (aus der Aufgabenstellung übernehmen!)
m_E = 80000   # kg - Endmasse der Rakete nach Brennphase (aus der Aufgabenstellung übernehmen!)
t_E = 190     # s - Dauer der Brennphase (aus der Aufgabenstellung übernehmen!)

# ================================
# HIER ENDET DER BEREICH, DEN DU ANPASSEN MUSST
# ================================

mu = (m_A - m_E) / t_E  # kg/s - berechneter Massenstrom

# Zeitdiskretisierung – je mehr Punkte, desto genauer
t_values = np.linspace(0, t_E, 10000)
dt = t_values[1] - t_values[0]

# Beschleunigung a(t)
def a(t):
    return v_rel * (mu / (m_A - mu * t)) - g

a_values = a(t_values)

# Trapezregel zur numerischen Integration
def trapez_integral(y_values, dt):
    return np.cumsum((y_values[:-1] + y_values[1:]) / 2 * dt)

# Geschwindigkeit v(t)
v_values = np.zeros_like(t_values)
v_values[1:] = trapez_integral(a_values, dt)

# Höhe h(t)
h_values = np.zeros_like(t_values)
h_values[1:] = trapez_integral(v_values, dt)

# Analytische Lösungen (für Vergleich), gemäss aufgabenstellung ändern
v_analytical = v_rel * np.log(m_A / (m_A - mu * t_values)) - g * t_values
h_analytical = (-v_rel * (m_A - mu * t_values) / mu * np.log(m_A / (m_A - mu * t_values))
                + v_rel * t_values - 0.5 * g * t_values**2)

# Plots
plt.figure()
plt.plot(t_values, a_values)
plt.title("Beschleunigung a(t)")
plt.xlabel("Zeit [s]")
plt.ylabel("Beschleunigung [m/s²]")
plt.grid()

plt.figure()
plt.plot(t_values, v_values, label='numerisch')
plt.plot(t_values, v_analytical, '--', label='analytisch')
plt.title("Geschwindigkeit v(t)")
plt.xlabel("Zeit [s]")
plt.ylabel("Geschwindigkeit [m/s]")
plt.legend()
plt.grid()

plt.figure()
plt.plot(t_values, h_values, label='numerisch')
plt.plot(t_values, h_analytical, '--', label='analytisch')
plt.title("Höhe h(t)")
plt.xlabel("Zeit [s]")
plt.ylabel("Höhe [m]")
plt.legend()
plt.grid()

plt.show()

# Ausgabe der Endwerte
print(f"v(t_E) = {v_values[-1]:.2f} m/s")  # Endgeschwindigkeit
print(f"h(t_E) = {h_values[-1]:.2f} m")    # Endhöhe
print(f"a(t_E) = {a_values[-1]/g:.2f} g")  # Endbeschleunigung in g
