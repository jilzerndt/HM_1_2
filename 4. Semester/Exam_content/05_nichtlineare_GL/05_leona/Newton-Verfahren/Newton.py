import numpy as np
import matplotlib.pyplot as plt

# === Parameter definieren ===
a = 2
b = 4
tol = 1e-8
max_iter = 20
x0 = np.array([2.0, -1.0])  # Startwert

# === Funktionensystem definieren ===
def F(x):
    x1, x2 = x
    f1 = ((x1 - 2)**2) / a + ((x2 - 1)**2) / b - 1
    f2 = 1 - x1**2 - x2**2
    return np.array([f1, f2])

# === Automatische Jacobi-Matrix (numerisch) ===
def J_numeric(x, h=1e-8):
    n = len(x)
    Fx = F(x)
    J = np.zeros((n, n))
    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += h
        J[:, i] = (F(x_perturbed) - Fx) / h
    return J

# === Newton-Verfahren ===
print("== Newton-Verfahren ==")
for k in range(max_iter):
    Fx = F(x0)
    norm_inf = np.linalg.norm(Fx, ord=np.inf)
    print(f"Iteration {k}: x = {x0}, ||F(x)||_∞ = {norm_inf:.2e}")
    if norm_inf < tol:
        break
    Jx = J_numeric(x0)
    delta = np.linalg.solve(Jx, -Fx)
    x0 = x0 + delta

# === Ergebnis ===
print(f"\nLösung x ≈ {x0}, f(x) ≈ {F(x0)}")

# === Teil c) Plot zur visuellen Bestimmung der Anzahl Lösungen ===
x_vals = np.linspace(-2.5, 2.5, 400)
y_vals = np.linspace(-2.5, 2.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Funktionen zum Plotten:
Z1 = ((X - 2)**2) / a + ((Y - 1)**2) / b - 1     # Ellipse
Z2 = 1 - X**2 - Y**2                             # Kreis

plt.figure(figsize=(8, 6))
cont1 = plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
cont2 = plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)

plt.title("Schnittpunkte von Ellipse und Kreis (Lösungen)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend([cont1.collections[0], cont2.collections[0]], ["f₁(x, y) = 0", "f₂(x, y) = 0"])
plt.scatter(*x0, color="green", label="Lösung x*", zorder=5)
plt.legend()
plt.axis("equal")
plt.show()
