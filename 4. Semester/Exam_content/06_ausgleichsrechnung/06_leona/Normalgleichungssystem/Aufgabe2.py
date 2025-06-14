import numpy as np

# Gegebene Daten
x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
y = np.array([0.54, 0.44, 0.28, 0.18, 0.12, 0.08], dtype=float)

# Transformation: Kehrwert von y
z = 1 / y  # z_i = 1 / y_i
x2 = x**2  # x_i^2

# Designmatrix für lineares Modell: z_i = a + b * x_i^2
A = np.column_stack((np.ones_like(x2), x2))  # Spalten: [1, x^2]

# Kleinste-Quadrate-Lösung berechnen
params, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
a, b = params

# Ausgabe der Ergebnisse
print(f"a ≈ {a:.5f}")
print(f"b ≈ {b:.5f}")

# Optionale Ausgabe der Ausgleichsfunktion
def fitted_function(x):
    return 1 / (a + b * x**2)

# Beispiel: Werte ausgeben für Vergleich
x_test = np.linspace(0, 5, 11)
y_fit = fitted_function(x_test)

print("\nFitted values:")
for xi, yi in zip(x_test, y_fit):
    print(f"x = {xi:.1f} → y = {yi:.5f}")
