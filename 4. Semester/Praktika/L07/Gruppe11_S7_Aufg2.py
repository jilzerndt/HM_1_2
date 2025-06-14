import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from numpy.linalg import norm, solve

x = np.array([2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5])
y = np.array([159.57209984, 159.8851819, 159.89378952, 160.30305273, 160.84630757, 160.94703969,
              161.56961845, 162.31468058, 162.32140561, 162.88880047, 163.53234609, 163.85817086, 
              163.55339958, 163.86393263, 163.90535931, 163.44385491])

# Definiere die Ansatzfunktion
def f(x, lam):
    return lam[0] + lam[1] * 10**(lam[2] + lam[3] * x) / (1 + 10**(lam[2] + lam[3] * x))

# Berechne den Residuenvektor r
def residual(lam):
    return y - f(x, lam)

# Berechne die Jacobi-Matrix
def jacobian(lam):
    n = len(x)
    J = np.zeros((n, 4))
    
    # Berechne die partiellen Ableitungen
    for i in range(n):
        # Partielle Ableitung lam0
        J[i, 0] = -1
        
        exp_term = 10**(lam[2] + lam[3] * x[i])
        denom = 1 + exp_term
        
        # Partielle Ableitung lam1
        J[i, 1] = -exp_term / denom
        
        # Partielle Ableitung lam2
        J[i, 2] = -lam[1] * exp_term * np.log(10) / (denom**2)
        
        # Partielle Ableitung lam3
        J[i, 3] = -lam[1] * exp_term * np.log(10) * x[i] / (denom**2)
    
    return J

# Berechne das Fehler
def error_func(lam):
    r = residual(lam)
    return 0.5 * np.sum(r**2)

# Implementierung gedämpftes gauss newton verfahren
def damped_gauss_newton(lam0, max_iter=100, tol=1e-6):
    lam = lam0.copy()
    errors = [error_func(lam)]
    lambdas = [lam.copy()]
    
    for k in range(max_iter):
        r = residual(lam)
        J = jacobian(lam)
        
        JTJ = J.T @ J
        JTr = J.T @ r
        delta = solve(JTJ, JTr)
        
        alpha = 1.0
        rho = 0.5
        c = 0.1
        
        current_error = error_func(lam)
        
        while error_func(lam - alpha * delta) > current_error - c * alpha * np.dot(JTr, delta) and alpha > 1e-10:
            alpha *= rho
        
        lam = lam - alpha * delta
        
        errors.append(error_func(lam))
        lambdas.append(lam.copy())
        
        # Konvergenzprüfung
        if norm(delta) < tol:
            break
    
    return lam, errors, lambdas, k+1

# Implementierung ungedämpftes gauss newton verfahren
def undamped_gauss_newton(lam0, max_iter=100, tol=1e-6):
    lam = lam0.copy()
    errors = [error_func(lam)]
    lambdas = [lam.copy()]
    
    for k in range(max_iter):
        r = residual(lam)
        J = jacobian(lam)
        
        JTJ = J.T @ J
        JTr = J.T @ r
        delta = solve(JTJ, JTr)
        
        lam = lam - delta
        
        errors.append(error_func(lam))
        lambdas.append(lam.copy())
        
        # Konvergenzprüfung
        if norm(delta) < tol:
            break
    
    return lam, errors, lambdas, k+1

# Funktion für fmin (scipy.optimize)
def error_for_fmin(lam):
    return error_func(lam)

print("Teil a: Gedämpftes Gauss-Newton-Verfahren")
lambda_start = np.array([100, 120, 3, -1])
lambda_opt_damped, errors_damped, lambdas_damped, iter_damped = damped_gauss_newton(lambda_start)

print(f"Optimale Parameter: lambda = {lambda_opt_damped}")
print(f"Anzahl Iterationen: {iter_damped}")
print(f"Finaler Fehler: {errors_damped[-1]}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Datenpunkte')

x_fine = np.linspace(min(x), max(x), 1000)
y_fit = f(x_fine, lambda_opt_damped)

plt.plot(x_fine, y_fit, color='red', label='Fit mit gedämpftem Gauss-Newton')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Datenfit mit gedämpftem Gauss-Newton-Verfahren')
plt.legend()
plt.grid(True)
plt.show()

print("\nTeil b: Ungedämpftes Gauss-Newton-Verfahren")
try:
    lambda_opt_undamped, errors_undamped, lambdas_undamped, iter_undamped = undamped_gauss_newton(lambda_start)
    print(f"Optimale Parameter: lambda = {lambda_opt_undamped}")
    print(f"Anzahl Iterationen: {iter_undamped}")
    print(f"Finaler Fehler: {errors_undamped[-1]}")
    
    converges = True
except Exception as e:
    print(f"Fehler beim ungedämpften Gauss-Newton: {e}")
    converges = False

plt.figure(figsize=(10, 6))
plt.semilogy(range(len(errors_damped)), errors_damped, 'r-', label='Gedämpftes Gauss-Newton')
if converges:
    plt.semilogy(range(len(errors_undamped)), errors_undamped, 'b-', label='Ungedämpftes Gauss-Newton')
plt.xlabel('Iteration')
plt.ylabel('Fehler (log-Skala)')
plt.title('Fehlerentwicklung')
plt.legend()
plt.grid(True)
plt.show()

print("\nTeil c: Optimierung mit scipy.optimize.fmin")
lambda_fmin = fmin(error_for_fmin, lambda_start, disp=True)
print(f"Optimale Parameter mit fmin: lambda = {lambda_fmin}")
print(f"Finaler Fehler mit fmin: {error_func(lambda_fmin)}")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Datenpunkte')

y_fit_damped = f(x_fine, lambda_opt_damped)
plt.plot(x_fine, y_fit_damped, color='red', label='Gedämpftes Gauss-Newton')

y_fit_fmin = f(x_fine, lambda_fmin)
plt.plot(x_fine, y_fit_fmin, color='green', label='scipy.optimize.fmin')

if converges:
    y_fit_undamped = f(x_fine, lambda_opt_undamped)
    plt.plot(x_fine, y_fit_undamped, color='orange', label='Ungedämpftes Gauss-Newton')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Vergleich der verschiedenen Fits')
plt.legend()
plt.grid(True)
plt.show()

"""
b) Konvergenz ungedämpftes gauss newton verfahren

Ja auch das ungedämpfte verfahren konvergiert. Es konvergiert langsamer und beim 2. Schritt
des verfahrens wird es zuerst ungenauer bevor es schlussendlich konvergiert.
"""
