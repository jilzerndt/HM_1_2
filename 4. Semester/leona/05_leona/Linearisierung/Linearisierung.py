import numpy as np
from sympy import symbols, Matrix, log, exp, diff, lambdify

# Symbole definieren
x1, x2, x3 = symbols('x1 x2 x3')

# Ursprungsfunktion f definieren
f1 = x1 + x2**2 - x3**2 - 13
f2 = log(x2 / 4) + exp(0.5 * x3) - 1
f3 = (x2 - 3)**2 - x3**3 + 7

f = Matrix([f1, f2, f3])

# Stelle für Linearisation
x0 = np.array([1.5, 3.0, 2.5])

# Jacobimatrix berechnen
vars = Matrix([x1, x2, x3])
J = f.jacobian(vars)

# In numerische Funktionen umwandeln
f_func = lambdify((x1, x2, x3), f, modules='numpy')
J_func = lambdify((x1, x2, x3), J, modules='numpy')

# Funktion und Jacobimatrix an x0 auswerten
f_val = f_func(*x0)
J_val = J_func(*x0)

# Linearisiertes Modell berechnen: g(x) = J(x0) * (x - x0) + f(x0)
def g_linear(x):
    x = np.array(x)
    delta_x = x - x0
    return np.dot(J_val, delta_x) + np.array(f_val, dtype=float)

# Test: Ausgabe der linearen Näherung an einem Punkt
test_x = np.array([1.5, 3.0, 2.5])  # sollte f(x0) ergeben
print("g(x0) =", g_linear(test_x))  # Test: sollte f(x0) entsprechen

# Optional: Ausdruck von g(x) symbolisch aufbauen
delta_x = Matrix([x1 - x0[0], x2 - x0[1], x3 - x0[2]]) # darf nicht angepasst werden!
g_expr = Matrix(f_val) + J_val @ delta_x
print("\nSymbolischer Ausdruck für g(x):")
print(g_expr)
