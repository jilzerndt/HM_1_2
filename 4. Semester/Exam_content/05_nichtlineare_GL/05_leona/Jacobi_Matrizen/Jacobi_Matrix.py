import sympy as sp

# Jacobi Matrice berechnet dei steigung einer Funktion an einer bestimmten Stelle

# Variablen definieren
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Funktionen für Aufgabe a)
f1_a = 5 * x1 * x2
f2_a = x1**2 * x2**2 + x1 + 2 * x2

# Jacobi-Matrix für Aufgabe a)
F_a = sp.Matrix([f1_a, f2_a])
vars_a = sp.Matrix([x1, x2])
J_a = F_a.jacobian(vars_a)

# Funktionen für Aufgabe b)
f1_b = sp.ln(x1**2 + x2**2) + x3**2
f2_b = sp.exp(x2**2 / (x2**2 + x3**2)) + x1**2
f3_b = (x3**3 + x2**2) / (x3**2 + x2**2 + x1**2)

# Jacobi-Matrix für Aufgabe b)
F_b = sp.Matrix([f1_b, f2_b, f3_b])
vars_b = sp.Matrix([x1, x2, x3])
J_b = F_b.jacobian(vars_b)

# Werte einsetzen für a) (x1, x2) = (1, 2)
J_a_num = J_a.subs({x1: 1, x2: 2})

# Werte einsetzen für b) (x1, x2, x3) = (1, 2, 3)
J_b_num = J_b.subs({x1: 1, x2: 2, x3: 3})

print("Jacobi-Matrix für Aufgabe a):")
sp.pprint(J_a)
print("\nEingesetzt bei (x1, x2) = (1, 2):")
sp.pprint(J_a_num)

print("\nJacobi-Matrix für Aufgabe b):")
sp.pprint(J_b)
print("\nEingesetzt bei (x1, x2, x3) = (1, 2, 3):")
sp.pprint(J_b_num)
