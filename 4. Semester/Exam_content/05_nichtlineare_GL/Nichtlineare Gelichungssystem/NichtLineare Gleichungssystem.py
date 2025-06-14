import numpy as np
from scipy.linalg import solve


def f(x):
    x1, x2, x3 = x
    return np.array([
        x1 + x2 ** 2 - x3 ** 2 - 13,
        np.log(x2 / 4) + np.exp(0.5 * x3 - 1) - 1,
        (x2 - 3) ** 2 - x3 ** 3 + 7
    ])


def jacobian(x):
    x1, x2, x3 = x
    return np.array([
        [1, 2 * x2, -2 * x3],
        [0, 1 / x2, 0.5 * np.exp(0.5 * x3 - 1)],
        [0, 2 * (x2 - 3), -3 * x3 ** 2]
    ])


def damped_newton(f, jacobian, x0, tol=1e-5, max_iter=100):
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        fx = f(x)
        if np.linalg.norm(fx, 2) < tol:
            return x
        J = jacobian(x)
        delta_x = solve(J, -fx)

        # Dämpfungsstrategie
        lambda_ = 1
        while np.linalg.norm(f(x + lambda_ * delta_x), 2) > (1 - 0.5 * lambda_) * np.linalg.norm(fx, 2):
            lambda_ *= 0.5

        x += lambda_ * delta_x

    raise ValueError("Newton-Verfahren hat die gewünschte Genauigkeit nicht erreicht.")


# Startwert
x0 = [1.5, 3, 2.5]

# Lösung berechnen
solution = damped_newton(f, jacobian, x0)
print("Lösung:", solution)
