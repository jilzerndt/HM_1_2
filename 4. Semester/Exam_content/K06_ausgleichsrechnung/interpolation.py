import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
import sympy as sp
from support_functions.plotting import plot_data_and_fit


def lagrange_interpolation(x_data, y_data, x_eval):
    """
    Lagrange interpolation for given data points
    
    Args:
        x_data: x-coordinates of data points
        y_data: y-coordinates of data points  
        x_eval: points where to evaluate interpolation
    
    Returns:
        y_eval: interpolated values at x_eval
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_eval = np.array(x_eval)
    
    n = len(x_data)
    
    if np.isscalar(x_eval):
        x_eval = np.array([x_eval])
    
    y_eval = np.zeros_like(x_eval, dtype=float)
    
    for i in range(len(x_eval)):
        result = 0.0
        for j in range(n):
            # Calculate Lagrange basis polynomial l_j(x)
            l_j = 1.0
            for k in range(n):
                if k != j:
                    l_j *= (x_eval[i] - x_data[k]) / (x_data[j] - x_data[k])
            result += y_data[j] * l_j
        y_eval[i] = result
    
    return y_eval if len(y_eval) > 1 else y_eval[0]

def quick_lagrange(x_data, y_data, x_eval):
    """Quick Lagrange interpolation"""
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    if np.isscalar(x_eval):
        x_eval = [x_eval]
    
    result = []
    for x in x_eval:
        y = 0
        for i in range(len(x_data)):
            li = 1
            for j in range(len(x_data)):
                if i != j:
                    li *= (x - x_data[j]) / (x_data[i] - x_data[j])
            y += y_data[i] * li
        result.append(y)
    
    return result[0] if len(result) == 1 else np.array(result)


def lagrange_manual(x_data, y_data, x_eval, show_steps=True):
    """
    Lagrange interpolation with detailed manual calculation steps
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data)
    
    if show_steps:
        print("Lagrange Interpolation")
        print("="*30)
        print(f"Data points: {list(zip(x_data, y_data))}")
        print(f"Evaluate at x = {x_eval}")
        print()
        print("Lagrange basis polynomials:")
    
    result = 0.0
    
    for i in range(n):
        # Calculate li(x_eval)
        li_val = 1.0
        li_str = f"l_{i}({x_eval}) = "
        
        for j in range(n):
            if i != j:
                factor = (x_eval - x_data[j]) / (x_data[i] - x_data[j])
                li_val *= factor
                li_str += f"({x_eval}-{x_data[j]})/({x_data[i]}-{x_data[j]}) × "
        
        li_str = li_str.rstrip(" × ")
        
        if show_steps:
            print(f"  {li_str} = {li_val:.6f}")
        
        contribution = y_data[i] * li_val
        result += contribution
        
        if show_steps:
            print(f"  Contribution: {y_data[i]} × {li_val:.6f} = {contribution:.6f}")
            print()
    
    if show_steps:
        print(f"P({x_eval}) = {result:.6f}")
    
    return result

def natural_cubic_spline_coefficients(x_data, y_data):
    """
    Calculate coefficients for natural cubic spline
    
    Args:
        x_data: x-coordinates of data points (sorted)
        y_data: y-coordinates of data points
    
    Returns:
        coefficients: dict with keys 'a', 'b', 'c', 'd' for each interval
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data) - 1  # number of intervals
    
    # Step 1: a_i = y_i
    a = y_data.copy()
    
    # Step 2: h_i = x_{i+1} - x_i
    h = np.diff(x_data)
    
    # Step 3: Set up tridiagonal system for c coefficients
    # Natural spline: c_0 = c_n = 0
    A = np.zeros((n-1, n-1))
    b = np.zeros(n-1)
    
    # Fill the tridiagonal matrix
    for i in range(n-1):
        if i == 0:
            A[i, i] = 2 * (h[0] + h[1])
            if n > 2:
                A[i, i+1] = h[1]
            b[i] = 3 * ((y_data[2] - y_data[1])/h[1] - (y_data[1] - y_data[0])/h[0])
        elif i == n-2:
            A[i, i-1] = h[i]
            A[i, i] = 2 * (h[i] + h[i+1])
            b[i] = 3 * ((y_data[i+2] - y_data[i+1])/h[i+1] - (y_data[i+1] - y_data[i])/h[i])
        else:
            A[i, i-1] = h[i]
            A[i, i] = 2 * (h[i] + h[i+1])
            A[i, i+1] = h[i+1]
            b[i] = 3 * ((y_data[i+2] - y_data[i+1])/h[i+1] - (y_data[i+1] - y_data[i])/h[i])
    
    # Solve for c coefficients
    if n > 2:
        c_inner = np.linalg.solve(A, b)
        c = np.zeros(n+1)
        c[1:n] = c_inner
    else:
        c = np.zeros(n+1)
    
    # Step 4: Calculate b and d coefficients
    b_coeff = np.zeros(n)
    d_coeff = np.zeros(n)
    
    for i in range(n):
        b_coeff[i] = (y_data[i+1] - y_data[i])/h[i] - h[i]/3 * (c[i+1] + 2*c[i])
        d_coeff[i] = (c[i+1] - c[i])/(3*h[i])
    
    return {
        'a': a,
        'b': b_coeff,
        'c': c,
        'd': d_coeff,
        'x': x_data
    }

def evaluate_cubic_spline(coeffs, x_eval):
    """
    Evaluate cubic spline at given points
    
    Args:
        coeffs: coefficients from natural_cubic_spline_coefficients
        x_eval: points where to evaluate spline
    
    Returns:
        y_eval: spline values at x_eval
    """
    x_data = coeffs['x']
    a, b, c, d = coeffs['a'], coeffs['b'], coeffs['c'], coeffs['d']
    
    x_eval = np.array(x_eval)
    if np.isscalar(x_eval):
        x_eval = np.array([x_eval])
    
    y_eval = np.zeros_like(x_eval)
    
    for j, x in enumerate(x_eval):
        # Find the interval
        i = 0
        for k in range(len(x_data)-1):
            if x_data[k] <= x <= x_data[k+1]:
                i = k
                break
        
        # Evaluate S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3
        dx = x - x_data[i]
        y_eval[j] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    
    return y_eval if len(y_eval) > 1 else y_eval[0]

def quick_spline(x_data, y_data, x_eval=None, plot=True):
    """Quick cubic spline interpolation"""
    if x_eval is None:
        x_eval = np.linspace(min(x_data), max(x_data), 100)
    
    # Use scipy's cubic spline
    cs = interpolate.CubicSpline(x_data, y_data, bc_type='natural')
    y_eval = cs(x_eval)
    
    if plot:
        plot_data_and_fit(x_data, y_data, x_eval, y_eval, "Cubic Spline Interpolation")
    
    return x_eval, y_eval


def cubic_spline_manual(x_data, y_data, show_coefficients=True):
    """
    Natural cubic spline with manual coefficient calculation
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data) - 1  # number of intervals
    
    print("Natural Cubic Spline Construction")
    print("="*40)
    print(f"Data points: {list(zip(x_data, y_data))}")
    print(f"Number of intervals: {n}")
    print()
    
    # Step 1: ai = yi
    a = y_data.copy()
    print("Step 1: Set ai = yi")
    for i in range(len(a)):
        print(f"  a{i} = {a[i]}")
    print()
    
    # Step 2: hi = xi+1 - xi
    h = np.diff(x_data)
    print("Step 2: Calculate hi = xi+1 - xi")
    for i in range(n):
        print(f"  h{i} = {x_data[i+1]} - {x_data[i]} = {h[i]}")
    print()
    
    # Step 3: Solve tridiagonal system for c coefficients
    print("Step 3: Solve tridiagonal system for c coefficients")
    print("Natural boundary conditions: c0 = cn = 0")
    
    if n >= 3:
        A = np.zeros((n-1, n-1))
        b_vec = np.zeros(n-1)
        
        # Build tridiagonal matrix
        for i in range(n-1):
            if i == 0:
                A[i, i] = 2 * (h[0] + h[1])
                if n > 2:
                    A[i, i+1] = h[1]
                b_vec[i] = 3 * ((y_data[2] - y_data[1])/h[1] - (y_data[1] - y_data[0])/h[0])
            elif i == n-2:
                A[i, i-1] = h[i]
                A[i, i] = 2 * (h[i] + h[i+1])
                b_vec[i] = 3 * ((y_data[i+2] - y_data[i+1])/h[i+1] - (y_data[i+1] - y_data[i])/h[i])
            else:
                A[i, i-1] = h[i]
                A[i, i] = 2 * (h[i] + h[i+1])
                A[i, i+1] = h[i+1]
                b_vec[i] = 3 * ((y_data[i+2] - y_data[i+1])/h[i+1] - (y_data[i+1] - y_data[i])/h[i])
        
        print("System matrix A:")
        print(A)
        print(f"Right-hand side b: {b_vec}")
        
        # Solve system
        c_inner = np.linalg.solve(A, b_vec)
        c = np.zeros(n+1)
        c[1:n] = c_inner
        
        print(f"Solution: c = {c}")
    else:
        c = np.zeros(n+1)
    
    print()
    
    # Step 4: Calculate b and d coefficients
    b_coeff = np.zeros(n)
    d_coeff = np.zeros(n)
    
    print("Step 4: Calculate b and d coefficients")
    for i in range(n):
        b_coeff[i] = (y_data[i+1] - y_data[i])/h[i] - h[i]/3 * (c[i+1] + 2*c[i])
        d_coeff[i] = (c[i+1] - c[i])/(3*h[i])
        
        if show_coefficients:
            print(f"  b{i} = {b_coeff[i]:.6f}")
            print(f"  d{i} = {d_coeff[i]:.6f}")
    
    print()
    print("Step 5: Spline functions Si(x) for each interval:")
    for i in range(n):
        print(f"  S{i}(x) = {a[i]:.4f} + {b_coeff[i]:.4f}(x-{x_data[i]}) + {c[i]:.4f}(x-{x_data[i]})² + {d_coeff[i]:.4f}(x-{x_data[i]})³")
        print(f"    for x ∈ [{x_data[i]}, {x_data[i+1]}]")
    
    return {
        'a': a, 'b': b_coeff, 'c': c, 'd': d_coeff,
        'x': x_data, 'intervals': n
    }

