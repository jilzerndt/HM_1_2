import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sympy as sp
import math

# custom import for trapezoidal rule
from int_rules import trapezoidal_rule

def romberg_extrapolation(f, a, b, m):
    """
    Romberg extrapolation for numerical integration
    
    Args:
        f: function to integrate
        a, b: integration limits
        m: number of extrapolation levels
    
    Returns:
        Romberg table (T matrix)
        Best approximation (T[m, m])
    """
    T = np.zeros((m + 1, m + 1))
    
    # First column: trapezoidal rule with h_j = (b-a)/2^j
    for j in range(m + 1):
        n = 2**j
        T[j, 0] = trapezoidal_rule(f, a, b, n)
    
    # Extrapolation
    for k in range(1, m + 1):
        for j in range(m - k + 1):
            T[j, k] = (4**k * T[j+1, k-1] - T[j, k-1]) / (4**k - 1)
    
    return T, T[0, m]


def quick_romberg_table(f_str, a, b, m=4):
    """
    Display Romberg extrapolation table
    
    Args:
        f_str: function as string
        a, b: integration limits
        m: number of levels
    """
    x = sp.Symbol('x')
    f = sp.lambdify(x, sp.sympify(f_str), 'numpy')
    
    T = np.zeros((m, m))
    
    # First column: trapezoidal rule
    for j in range(m):
        n = 2**j
        h = (b - a) / n
        x_vals = np.linspace(a, b, n + 1)
        y_vals = f(x_vals)
        T[j, 0] = h * (0.5 * (y_vals[0] + y_vals[-1]) + np.sum(y_vals[1:-1]))
    
    # Extrapolation
    for k in range(1, m):
        for j in range(m - k):
            T[j, k] = (4**k * T[j+1, k-1] - T[j, k-1]) / (4**k - 1)
    
    # Display table
    print(f"Romberg Table for integral of {f_str} from {a} to {b}:")
    print("j\\k", end="")
    for k in range(m):
        print(f"        T[j,{k}]", end="")
    print()
    
    for j in range(m):
        print(f"{j:2d} ", end="")
        for k in range(m-j):
            print(f"{T[j,k]:12.8f}", end="")
        print()
    
    return T


def romberg_manual(f_str, a, b, m=4, show_table=True):
    """
    Romberg extrapolation with complete table display
    """
    x_sym = sp.Symbol('x')
    f_expr = sp.sympify(f_str)
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    
    print(f"Romberg Extrapolation")
    print("="*30)
    print(f"Function: f(x) = {f_str}")
    print(f"Interval: [{a}, {b}]")
    print(f"Levels: m = {m}")
    print()
    
    T = np.zeros((m + 1, m + 1))
    
    # First column: Trapezoidal rule with halving step sizes
    print("Step 1: Calculate T[j,0] using trapezoidal rule")
    for j in range(m + 1):
        n = 2**j
        h = (b - a) / n
        x_vals = np.linspace(a, b, n + 1)
        y_vals = f_func(x_vals)
        T[j, 0] = h * (0.5 * (y_vals[0] + y_vals[-1]) + np.sum(y_vals[1:-1]))
        
        if show_table:
            print(f"  j={j}, n=2^{j}={n:2d}, h={h:.4f}, T[{j},0] = {T[j,0]:.8f}")
    print()
    
    # Extrapolation
    print("Step 2: Richardson extrapolation")
    for k in range(1, m + 1):
        for j in range(m - k + 1):
            T[j, k] = (4**k * T[j+1, k-1] - T[j, k-1]) / (4**k - 1)
    
    # Display table
    if show_table:
        print("\nRomberg Table:")
        print("j\\k", end="")
        for k in range(m + 1):
            print(f"        T[j,{k}]", end="")
        print()
        print("-" * (12 * (m + 2)))
        
        for j in range(m + 1):
            print(f"{j:2d} ", end="")
            for k in range(m - j + 1):
                print(f"{T[j,k]:12.8f}", end="")
            print()
    
    print(f"\nBest approximation: T[0,{m}] = {T[0,m]:.10f}")
    return T