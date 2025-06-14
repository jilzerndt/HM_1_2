import numpy as np

# Funktionen definieren
def f(vec):
    x, y = vec
    return np.array([
        c * x + y - 4,
        x**2 + y**2 - 9
    ])

# Jacobimatrix
def jacobian(vec):
    x, y = vec
    return np.array([
        [c, 1],
        [2 * x, 2 * y]
    ])

# Parameter c und Startwert
c = 1
x = np.array([0.0, 3.0])
tolerance = 1e-4
max_iter = 100
iterations = 0

# Exakte Lösung (nur zur Fehlerberechnung)
exact_x = np.array([
    (4 * c - np.sqrt(9 * c**2 - 7)) / (c**2 + 1),
    (c * np.sqrt(9 * c**2 - 7) - 4 * c**2) / (c**2 + 1) + 4
])

# Newton-Verfahren
while iterations < max_iter:
    fx = f(x)
    J = jacobian(x)
    delta = np.linalg.solve(J, -fx)
    x = x + delta
    iterations += 1
    # Abbruchbedingung: maximaler absoluter Fehler
    if np.max(np.abs(x - exact_x)) <= tolerance:
        break

iterations, x
print("Anzahl iterationen:", iterations)
print(x)
