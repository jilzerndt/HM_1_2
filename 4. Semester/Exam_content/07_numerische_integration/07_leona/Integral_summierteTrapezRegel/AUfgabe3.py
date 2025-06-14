import numpy as np

def Name_S8_Aufg3a(x, y):
    """
    Summierte Trapezregel für nicht äquidistante x-Werte.
    Parameters:
        x : array-like, x-Werte (z. B. Radius in m)
        y : array-like, Funktionswerte f(x)
    Returns:
        Integralwert über das Intervall
    """
    integral = 0.0
    for i in range(len(x) - 1):
        integral += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return integral

# --- Teil b: Erdmasse berechnen ---

# Gegebene Daten (r in km, ρ in kg/m³)
r_km = np.array([0, 800, 1200, 1400, 2000, 3000, 3400, 3600, 4000, 5000, 5500, 6370])
rho =  np.array([13000, 12900, 12700, 12000, 11650, 10600, 9900, 5500, 5300, 4750, 4500, 3300])

# Umrechnen von km in m
r_m = r_km * 1_000

# Integrand: f(r) = ρ(r) * 4πr²
f_r = rho * 4 * np.pi * r_m**2

# Integration mit Trapezregel
m_berechnet = Name_S8_Aufg3a(r_m, f_r)

# Literaturwert (aus der Aufgabenlösung)
m_literatur = 5.976e24  # [kg]

# Fehlerberechnung
abs_fehler = abs(m_berechnet - m_literatur)
rel_fehler = abs_fehler / m_literatur

# Ausgabe
print(f"Berechnete Erdmasse:       {m_berechnet:.3e} kg")
print(f"Literaturwert Erdmasse:    {m_literatur:.3e} kg")
print(f"Absoluter Fehler:          {abs_fehler:.3e} kg")
print(f"Relativer Fehler:          {rel_fehler:.3f}")
