import numpy as np

# === Teil a) Parameterschätzung mit Newton-Verfahren ===

# Gegebene 3 Messpunkte (t_i, g_i) – aus Grafik geschätzt
t_data = np.array([1.0, 1.6, 2.0])         # Zeitpunkte in Stunden
g_data = np.array([40.0, 250.0, 800.0])    # Anzahl in Mio Bakterien

# Startwert für Parametervektor x = [a, b, c]
x0 = np.array([1.0, 2.0, 3.0])

# Funktionsausdruck F(x): Residuenvektor
def F(x):
    a, b, c = x
    return np.array([
        a + b * np.exp(c * t_data[0]) - g_data[0],
        a + b * np.exp(c * t_data[1]) - g_data[1],
        a + b * np.exp(c * t_data[2]) - g_data[2]
    ], dtype=float)

# Jacobi-Matrix Df(x)
def J(x):
    a, b, c = x
    return np.array([
        [1, np.exp(c * t_data[0]), b * t_data[0] * np.exp(c * t_data[0])],
        [1, np.exp(c * t_data[1]), b * t_data[1] * np.exp(c * t_data[1])],
        [1, np.exp(c * t_data[2]), b * t_data[2] * np.exp(c * t_data[2])]
    ], dtype=float)

# Berechne erste Iteration
Fx0 = F(x0)
Jx0 = J(x0)
delta = np.linalg.solve(Jx0, -Fx0)
x1 = x0 + delta

print("=== Teil a) Erste Newton-Iteration ===")
print(f"Startwert x0      = {np.round(x0, 4)}")
print(f"Residual F(x0)    = {np.round(Fx0, 4)}")
print(f"Jacobi-Matrix J(x0):\n{np.round(Jx0, 4)}")
print(f"delta             = {np.round(delta, 4)}")
print(f"x1 (neuer Wert)   = {np.round(x1, 4)}")

# Werte für Teil b:
a, b, c = x1  # = (-5.3933, 2.5236, 2.8681)

# === Teil b) Bestimme t für g(t) = 1600 ===

# Definition der Funktion g(t)
def g(t):
    return a + b * np.exp(c * t)

# Zielwert
ZIEL = 1600

# f(t) = g(t) - 1600, f'(t) = g'(t)
def f(t):
    return g(t) - ZIEL

def f_prime(t):
    return b * c * np.exp(c * t)

# Newton-Verfahren für Skalar
t0 = 2.2     # Startwert
tol = 1e-4
max_iter = 20

print("\n=== Teil b) Newton-Verfahren zur Bestimmung von t ===")
for i in range(max_iter):
    f_val = f(t0)
    fp_val = f_prime(t0)
    t1 = t0 - f_val / fp_val
    print(f"t_{i+1} = {t1:.6f}")
    if abs(t1 - t0) < tol:
        break
    t0 = t1

print(f"\nErgebnis: t ≈ {t1:.6f} h, g(t) ≈ {g(t1):.2f} Mio Bakterien")
