import numpy as np
import matplotlib.pyplot as plt

# Daten aus der Tabelle
years = np.array([1971, 1972, 1974, 1978, 1982, 1985, 1989, 1993, 1997, 1999, 2000, 2002, 2003])
transistors = np.array([2250, 2500, 5000, 29000, 120000, 275000, 1180000, 3100000,
                        7500000, 24000000, 42000000, 220000000, 410000000])

# x = Jahr - 1970 (damit einfacher mit θ2)
x = years - 1970
y = np.log10(transistors)

# Lineare Regression: log10(N) = θ1 + θ2 * (t - 1970)
A = np.vstack([np.ones_like(x), x]).T
theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
theta1, theta2 = theta

# Fitfunktion
x_fit = np.linspace(min(x), 2015 - 1970, 500)
y_fit = theta1 + theta2 * x_fit
transistor_fit = 10**y_fit

# Plot
plt.figure(figsize=(8, 6))
plt.semilogy(years, transistors, 'o', label='Messdaten')
plt.semilogy(x_fit + 1970, transistor_fit, '-', label='Fit: log10(N) = θ1 + θ2·(t - 1970)')
plt.xlabel('Jahr')
plt.ylabel('Anzahl Transistoren / Chip')
plt.title('Moore’sches Gesetz – Halblogarithmische Darstellung')
plt.grid(True, which='both')
plt.legend()
plt.show()

# Extrapolation: Vorhersage für Jahr 2015
year_2015 = 2015 - 1970
log_N_2015 = theta1 + theta2 * year_2015
N_2015 = 10**log_N_2015

print(f"θ1 = {theta1:.4f}")
print(f"θ2 = {theta2:.4f}")
print(f"Vorhersage für 2015: {N_2015:.2e} Transistoren (tatsächlich ~4e9)")

# Moore'sches Gesetz Vergleich
# N verdoppelt sich alle T Jahre -> log10(N) steigt um log10(2) ≈ 0.3010 pro T Jahre
T_verdopplung = 0.3010 / theta2
print(f"Verdopplungszeit laut Fit: {T_verdopplung:.2f} Jahre (Moore: 1.5–2 Jahre)")
