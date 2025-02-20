import numpy as np
import matplotlib.pyplot as plt
import math


#Initialisierungswerte aus Aufgabenstellung
tol = 10**(-18)
a = -1
b = 3
I = np.array[a, b]


def F(x):
    return (np.e**x + np.e**(-x) - 4)/3

"""Aufgabe 4 a)"""

x0 = 2.0

def fixed_point_iteration(F, x0, tol, max_iter=100, store_history=False):
    """
    Perform fixed-point iteration x_{n+1} = g(x_n).
    
    Args:
        g: Fixed-point function
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        store_history: Whether to store iteration history
        
    Returns:
        Tuple of (final approximation, iterations performed, history if requested)
    """
    x_prev = x0
    history = [x0] if store_history else None
    
    for i in range(max_iter):
        # Iteration
        x_next = F(x_prev)
        
        #Ausgerechnete Werte speichern
        if store_history:
            history.append(x_next)
        
        #Abbruchkriterium
        if abs(x_next - x_prev) < tol:
            return x_next, i + 1, history
        
        #weiter zur nächsten Iteration
        x_prev = x_next
        
    return x_next, max_iter, history

def plot_iteration_process(g, x0, a, b, num_iterations=5):
    """
    Visualize the fixed-point iteration process.
    
    Args:
        g: Fixed-point function
        x0: Initial guess
        a, b: Plot range
        num_iterations: Number of iterations to show
    """
    x = np.linspace(a, b, 1000)
    y_g = [g(xi) for xi in x]
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, y_g, 'b-', label='g(x)')
    plt.plot(x, x, 'k--', label='y=x')
    
    x_curr = x0
    plt.plot([x_curr], [x_curr], 'go', label='Start')
    
    for i in range(num_iterations):
        y_curr = g(x_curr)
        # Plot vertical line to g(x)
        plt.plot([x_curr, x_curr], [x_curr, y_curr], 'r:', alpha=0.5)
        # Plot horizontal line to y=x
        plt.plot([x_curr, y_curr], [y_curr, y_curr], 'r:', alpha=0.5)
        x_curr = y_curr
        plt.plot([x_curr], [x_curr], 'ro')
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fixed-Point Iteration Process')
    plt.legend()
    plt.axis('equal')
    plt.show()
    
"""Aufgabe 4 b)"""

x0 = 1.9
x1 = 2.1

def f(x): 
    return (np.e**x + np.e**(-x) - 4)/3

def secant_method(f, x0, x1, tol=1e-10, max_iter=100, store_history=False):
    """
    Standard secant method for finding roots: x_{n+1} = x_n - f(x_n)(x_n - x_{n-1})/(f(x_n) - f(x_{n-1}))
    Uses linear interpolation between two points to approximate derivative.
    
    Args:
        f: Function to find root of
        x0, x1: Initial guesses
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        store_history: Whether to store iteration history
    
    Returns:
        Tuple of (final approximation, iterations performed, history if requested)
    """
    # Check initial points aren't too close together
    if abs(f(x1) - f(x0)) < tol:
        raise ValueError("Initial points too close together")
        
    history = [x0, x1] if store_history else None
    
    for i in range(max_iter):
        f_x0, f_x1 = f(x0), f(x1)
        
        # Secant step using linear interpolation
        x_new = x1 - f_x1 * (x1 - x0)/(f_x1 - f_x0)
        
        if store_history:
            history.append(x_new)
            
        # Abbruchkriterium
        if abs(x_new) < tol:
            return x_new, i + 1, history
            
        # Update points for next iteration
        x0, x1 = x1, x_new
        
    return x1, max_iter, history