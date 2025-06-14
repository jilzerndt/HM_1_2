import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from sympy import symbols, diff, lambdify

# === Gegebene Messdaten ===
d = np.array([500, 1000, 1500, 2500, 3500, 4000, 4500, 5000, 5250, 5500])  # Drehzahlen
P = np.array([10.5, 49.2, 72.1, 85.4, 113, 121, 112, 80.2, 61.1, 13.8])    # Leistungen

# === a) Punktplot + Grad des Polynoms wählen ===
plt.figure(figsize=(8, 5))
plt.scatter(d, P, color='red', label='Messdaten')
plt.xlabel("Drehzahl [U/min]")
plt.ylabel("Leistung [kW]")
plt.title("Messdaten")
plt.grid(True)
plt.legend()
plt.show()

# Kommentar:
# Nach visuellem Eindruck scheint ein Polynom vom Grad 4 oder 5 sinnvoll.
# Du kannst das unten bei "grad" ändern:

grad = 4  # <- anpassen, wenn du anderen Grad willst

# === b) Normale Gleichungen manuell lösen (mit Skalierung) ===

# Skalierung der x-Werte für bessere Numerik
d_scaled = d / 1000  # <- andere Skalierung hier möglich

# Design-Matrix A aufbauen
A = np.vander(d_scaled, grad + 1, increasing=True)  # z.B. [1, x, x², ..., x^n]

# Normalgleichung: (AᵀA)·c = Aᵀ·P
ATA = A.T @ A
ATP = A.T @ P

# Lösen des LGS
coeffs = solve(ATA, ATP)  # enthält die Koeffizienten des Ausgleichspolynoms

# Kommentar:
# Bei anderer Aufgabe musst du nur:
# - d, P anpassen
# - evtl. andere Skalierung wählen
# - grad ändern

# Polynomauswertung für Plot
d_plot = np.arange(500, 5501, 1)
d_plot_scaled = d_plot / 1000
A_plot = np.vander(d_plot_scaled, grad + 1, increasing=True)
P_fit = A_plot @ coeffs

# Plot mit Ausgleichspolynom
plt.figure(figsize=(10, 6))
plt.scatter(d, P, color='red', label='Messdaten')
plt.plot(d_plot, P_fit, label=f'Ausgleichspolynom Grad {grad}', color='blue')
plt.xlabel("Drehzahl [U/min]")
plt.ylabel("Leistung [kW]")
plt.title("Ausgleichspolynom + Messdaten")
plt.grid(True)
plt.legend()
plt.show()

# === c) Newton-Verfahren zur Bestimmung des Maximums ===

# Symbolisches Polynom aufstellen
x = symbols('x')  # x ist skalierte Drehzahl
poly_expr = sum(coeffs[i] * x**i for i in range(grad + 1))
poly_prime = diff(poly_expr, x)
poly_doubleprime = diff(poly_prime, x)

# Funktionen für Newton-Verfahren
f = lambdify(x, poly_prime)
f_prime = lambdify(x, poly_doubleprime)

# Startwert: z.B. in der Nähe der maximalen Messwertstelle
x0 = 4.0  # Startwert (entspricht 4000 U/min)

tol = 1e-6
max_iter = 20

for i in range(max_iter):
    fx = f(x0)
    fpx = f_prime(x0)
    if abs(fpx) < 1e-10:
        print("Ableitung zu klein – Verfahren abgebrochen.")
        break
    x1 = x0 - fx / fpx
    if abs(x1 - x0) < tol:
        break
    x0 = x1

max_drehzahl = x1 * 1000  # Rückskalierung
max_leistung = sum(coeffs[i] * (x1)**i for i in range(grad + 1))

print(f"\nMaximum der Leistung bei Drehzahl ≈ {max_drehzahl:.2f} U/min")
print(f"Maximale Leistung ≈ {max_leistung:.2f} kW")
