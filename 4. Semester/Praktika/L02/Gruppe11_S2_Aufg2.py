import sympy as sp

def compute_jacobian():
    x1, x2, x3 = sp.symbols('x1 x2 x3')

    # Funktionen aus Aufgabe 1
    f1 = sp.ln(x1 ** 2 + x2 ** 2) + x3 ** 2
    f2 = sp.exp(x2 ** 2 + x3 ** 2) + x1 ** 2
    f3 = 1 / (x3 ** 2 + x1 ** 2) + x2 ** 2

    # Funktionsvektor
    f = sp.Matrix([f1, f2, f3])

    # Variablenvektor
    X = sp.Matrix([x1, x2, x3])

    # Jacobi-Matrix
    J = f.jacobian(X)

    # Punkt für Evaluation
    point = {x1: 1, x2: 2, x3: 3}
    J_evaluated = J.subs(point)

    print("Jacobi-Matrix:")
    sp.pprint(J)
    print("\nEvaluierte Jacobi-Matrix bei (1,2,3):")
    sp.pprint(J_evaluated)


def linearize_function():
    x1, x2, x3 = sp.symbols('x1 x2 x3')

if __name__ == "__main__":
    print("Aufgabe 2: Jacobi-Matrix berechnen")
    compute_jacobian()

