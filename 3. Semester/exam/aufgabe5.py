import numpy as np
import matplotlib.pyplot as plt
import math

"""Aufgabe 5 Initialisierung der Matrix und Vektoren"""

A = np.array[[7, -2, -2], [-2, 7, -2], [-2, -2, 7]]
b = np.transpose(np.array[5, -13, 14])
x0 = np.transpose(np.array[0, 0, 0])

"""Aufgabe 5 a)"""

#impl. Iterationsfunktion
#herausfinden was mit der Aufgabe gemeint ist help

def calculate_a_posteriori_error(x_approx, x_exact):
    """Berechnet den a-posteriori Fehler ||x^(k) - x||."""
    error = np.linalg.norm(x_approx - x_exact, np.inf)
    return error
    

def fixed_point_iteration_with_error(g, x0, tol=10**(-5), max_iter=1000):
    """
    Fixed-point iteration with a priori and a posteriori error estimates.
    
    Args:
        g: Fixed-point function
        x0: Initial guess
        alpha: Lipschitz constant
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        
    Returns:
        Tuple of (final approximation, i = nr of iterations)
    """
    x_prev = x0
    x_curr = g(x0)
    
    first_step_size = abs(x_curr - x_prev)
    
    for i in range(max_iter):
        # Calculate error estimate
        a_posteriori = (x_curr, x_prev)
        
        if a_posteriori < tol:
            break
            
        x_prev = x_curr
        x_curr = g(x_curr)
    
    return x_curr, i

def gauss_seidel_iteration(A, b, x0, max_iter=1000, tol=1e-10):
    """
    Solve linear system using Gauss-Seidel iteration.
    x_{k+1} = (D+L)^{-1}(b - Ux_k) where A = D + L + U
    """
    A = np.array(A)
    b = np.array(b)
    x = np.array(x0)
    n = len(b)
    residual_norms = []
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        # Update each component using the latest values
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - 
                    np.dot(A[i,i+1:], x_old[i+1:])) / A[i,i]
        
        # Check convergence
        if np.max(np.abs(x - x_old)) < tol:
            return x, iteration + 1, residual_norms
            
        # Compute residual norm
        residual = np.linalg.norm(b - np.dot(A, x))
        residual_norms.append(residual)
        
        return x, max_iter, residual_norms