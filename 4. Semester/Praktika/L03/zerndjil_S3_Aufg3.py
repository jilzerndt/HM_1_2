import numpy as np
import scipy.linalg as la
import sympy as sp


def f(x):
    x1, x2, x3 = x
    return np.array([
        x1 + x2 ** 2 - x3 ** 2 - 13,
        np.log(x2 / 4) + np.exp(0.5 * x3 - 1) - 1,
        (x2 - 3) ** 2 - x3 ** 3 + 7
    ])


x1, x2, x3 = sp.symbols('x1 x2 x3')
f_sym = sp.Matrix([
    x1 + x2 ** 2 - x3 ** 2 - 13,
    sp.ln(x2 / 4) + sp.exp(0.5 * x3 - 1) - 1,
    (x2 - 3) ** 2 - x3 ** 3 + 7
])
J_sym = f_sym.jacobian([x1, x2, x3])
J_func = sp.lambdify((x1, x2, x3), J_sym, 'numpy')


def jacobian(x):
    return np.array(J_func(*x), dtype=float)


def damped_newton(f, jacobian, x0, tol=1e-5, max_iter=100):
    x_k = np.array(x0, dtype=float)
    for _ in range(max_iter):
        F_val = f(x_k)
        J_val = jacobian(x_k)
        delta = la.solve(J_val, -F_val)

        lambda_k = 1.0
        while np.linalg.norm(f(x_k + lambda_k * delta), 2) > (1 - 0.5 * lambda_k) * np.linalg.norm(F_val, 2):
            lambda_k *= 0.5
            if lambda_k < 1e-4:
                break

        x_k += lambda_k * delta

        if np.linalg.norm(F_val, 2) < tol:
            return x_k

    raise ValueError("Newton's method did not converge.")


x0 = [1.5, 3, 2.5]
solution = damped_newton(f, jacobian, x0)

print(f"Solution: x1 = {solution[0]:.6f}, x2 = {solution[1]:.6f}, x3 = {solution[2]:.6f}")

# Lösung: x1 = 1.000000, x2 = 4.000000, x3 = 3.000000
