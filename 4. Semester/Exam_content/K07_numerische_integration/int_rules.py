import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate
import math
from scipy import interpolate
import pandas as pd



def rectangle_rule(f, a, b, n):
    """
    Summated rectangle rule (midpoint rule)
    
    Args:
        f: function to integrate
        a, b: integration limits
        n: number of subintervals
    
    Returns:
        approximation of integral
    """
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)  # midpoints
    return h * np.sum(f(x))

def trapezoidal_rule(f, a, b, n):
    """
    Summated trapezoidal rule
    
    Args:
        f: function to integrate
        a, b: integration limits
        n: number of subintervals
    
    Returns:
        approximation of integral
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))

def trapezoidal_rule_non_equidistant(x_data, y_data):
    """
    Trapezoidal rule for non-equidistant data points
    
    Args:
        x_data: x-coordinates (must be sorted)
        y_data: y-coordinates
    
    Returns:
        approximation of integral
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    integral = 0.0
    for i in range(len(x_data) - 1):
        h = x_data[i+1] - x_data[i]
        integral += h * (y_data[i] + y_data[i+1]) / 2
    
    return integral

def simpson_rule(f, a, b, n):
    """
    Summated Simpson's rule
    
    Args:
        f: function to integrate
        a, b: integration limits
        n: number of subintervals (must be even)
    
    Returns:
        approximation of integral
    """
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Simpson's rule: h/3 * (y0 + 4*y1 + 2*y2 + 4*y3 + ... + 4*yn-1 + yn)
    result = y[0] + y[-1]  # endpoints
    result += 4 * np.sum(y[1::2])  # odd indices (4 coefficient)
    result += 2 * np.sum(y[2:-1:2])  # even indices (2 coefficient)
    
    return h * result / 3

def adaptive_simpson(f, a, b, tol=1e-6, max_depth=10):
    """
    Adaptive Simpson's rule with error control
    
    Args:
        f: function to integrate
        a, b: integration limits
        tol: tolerance
        max_depth: maximum recursion depth
    
    Returns:
        approximation of integral
    """
    def simpson_basic(f, a, b):
        """Basic Simpson's rule for interval [a, b]"""
        h = (b - a) / 2
        return h / 3 * (f(a) + 4 * f(a + h) + f(b))
    
    def adaptive_step(f, a, b, tol, depth):
        if depth > max_depth:
            return simpson_basic(f, a, b)
        
        c = (a + b) / 2
        
        # Calculate Simpson's rule for whole interval and two halves
        S = simpson_basic(f, a, b)
        S1 = simpson_basic(f, a, c)
        S2 = simpson_basic(f, c, b)
        
        # Error estimate
        error = abs(S1 + S2 - S) / 15
        
        if error < tol:
            return S1 + S2
        else:
            # Subdivide further
            left = adaptive_step(f, a, c, tol/2, depth+1)
            right = adaptive_step(f, c, b, tol/2, depth+1)
            return left + right
    
    return adaptive_step(f, a, b, tol, 0)

def quick_integrate(f_str, a, b, method='simpson', n=100):
    """
    Quick numerical integration
    
    Args:
        f_str: function as string expression
        a, b: integration limits
        method: 'rectangle', 'trapezoidal', 'simpson', 'romberg'
        n: number of intervals
    
    Example:
        result = quick_integrate("x**2", 0, 1, 'simpson', 100)
    """
    x = sp.Symbol('x')
    f = sp.lambdify(x, sp.sympify(f_str), 'numpy')
    
    if method == 'rectangle':
        h = (b - a) / n
        x_vals = np.linspace(a + h/2, b - h/2, n)
        result = h * np.sum(f(x_vals))
    
    elif method == 'trapezoidal':
        h = (b - a) / n
        x_vals = np.linspace(a, b, n + 1)
        y_vals = f(x_vals)
        result = h * (0.5 * (y_vals[0] + y_vals[-1]) + np.sum(y_vals[1:-1]))
    
    elif method == 'simpson':
        if n % 2 != 0:
            n += 1  # Make n even
        h = (b - a) / n
        x_vals = np.linspace(a, b, n + 1)
        y_vals = f(x_vals)
        result = h/3 * (y_vals[0] + y_vals[-1] + 4*np.sum(y_vals[1::2]) + 2*np.sum(y_vals[2:-1:2]))
    
    elif method == 'romberg':
        # Simple Romberg with 4 levels
        T = np.zeros((4, 4))
        for j in range(4):
            n_j = 2**j
            h = (b - a) / n_j
            x_vals = np.linspace(a, b, n_j + 1)
            y_vals = f(x_vals)
            T[j, 0] = h * (0.5 * (y_vals[0] + y_vals[-1]) + np.sum(y_vals[1:-1]))
        
        for k in range(1, 4):
            for j in range(4-k):
                T[j, k] = (4**k * T[j+1, k-1] - T[j, k-1]) / (4**k - 1)
        
        result = T[0, 3]
    
    print(f"Integral of {f_str} from {a} to {b} using {method}: {result:.8f}")
    return result