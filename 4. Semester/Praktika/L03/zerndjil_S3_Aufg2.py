import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

'''Grundinformationen aus Aufgabenstellung'''

# code aus aufgabenstellung, unverändert! (aufgabe a)
x, y = sp.symbols('x y')
f1 = (x**2 / 186**2) - (y**2 / (300**2 - 186**2)) - 1
f2 = ((y - 500)**2 / 279**2) - ((x - 300)**2 / (500**2 - 279**2)) - 1

p1 = sp.plot_implicit(sp.Eq(f1, 0), (x, -2000, 2000), (y, -2000, 2000), show=False, line_color='blue')
p2 = sp.plot_implicit(sp.Eq(f2, 0), (x, -2000, 2000), (y, -2000, 2000), show=False, line_color='red')
p1.append(p2[0])
p1.show()

# Von Auge geschätzt:
initial_guesses = [
    [-1300, 1600],
    [-200, 50],
    [200, 200],
    [750, 900],
]


'''Aufgabe b)'''

# benutze die oben bestimmten Näherungsvektoren und bestimme mit dem Newton-Verfahren die vier lösungen mit einer Genauigkeit ||f(x)^(k)||_2 < 10^-5

# Funktionen (f1 und f2 bleiben gleich)
F = sp.Matrix([f1, f2])
vars = sp.Matrix([x, y])

# Jacobi
J = F.jacobian(vars)
f_lambdified = sp.lambdify((x, y), F, 'numpy')
J_lambdified = sp.lambdify((x, y), J, 'numpy')

# Newton-Verfahren
def newton(F, J, x0, tol=1e-5, max_iter=50):
    x = x0
    for i in range(max_iter):
        F_val = np.array(F(*x)).astype(np.float64).flatten()
        if np.linalg.norm(F_val) < tol:
            break
        J_val = np.array(J(*x)).astype(np.float64)
        delta = np.linalg.solve(J_val, F_val)
        x = x - delta
    return x

# Lösungen
solutions = []
for guess in initial_guesses:
    solution = newton(f_lambdified, J_lambdified, guess)
    solutions.append(solution)
    
# Ausgabe (in Englisch weil Umlaute nicht funktionieren)
print("Solutions:")
for i, solution in enumerate(solutions):
    print(f"Solution {i+1}: {solution}")
    
    


