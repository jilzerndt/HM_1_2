"""
Chapter 5: Numerical Solution of Nonlinear Systems

"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import interpolate
import pandas as pd


# Import custom modules - for exam obviously copy and paste!!
from support_functions.basics import evaluate_at_point
from support_functions.plotting import plot_system_2d
from partielle_ableitungen import jacobian_symbolic, linearize_function



def newton_method_systems(f_func, df_func, x0, tol=1e-5, max_iter=50):
    """
    Newton's method for systems of nonlinear equations
    
    Args:
        f_func: function that returns f(x) as numpy array
        df_func: function that returns Jacobian matrix at x
        x0: initial guess as numpy array
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    
    Returns:
        tuple: (solution, iterations, convergence_info)
    """
    x = np.copy(x0)
    convergence_info = []
    
    for i in range(max_iter):
        f_val = f_func(x)
        df_val = df_func(x)
        
        # Check if Jacobian is singular
        if np.linalg.det(df_val) == 0:
            print(f"Warning: Singular Jacobian at iteration {i}")
            break
        
        # Newton step: solve J * delta = -f
        try:
            delta = np.linalg.solve(df_val, -f_val)
        except np.linalg.LinAlgError:
            print(f"Linear algebra error at iteration {i}")
            break
        
        x_new = x + delta
        
        # Store convergence information
        f_norm = np.linalg.norm(f_val, 2)
        delta_norm = np.linalg.norm(delta, 2)
        convergence_info.append({
            'iteration': i,
            'x': np.copy(x),
            'f_norm': f_norm,
            'delta_norm': delta_norm
        })
        
        # Check convergence
        if f_norm < tol:
            print(f"Converged after {i+1} iterations")
            return x_new, i+1, convergence_info
        
        x = x_new
    
    print(f"Did not converge after {max_iter} iterations")
    return x, max_iter, convergence_info

def damped_newton_method(f_func, df_func, x0, tol=1e-5, max_iter=50, k_max=4):
    """
    Damped Newton's method for systems of nonlinear equations
    
    Args:
        f_func: function that returns f(x) as numpy array
        df_func: function that returns Jacobian matrix at x
        x0: initial guess as numpy array
        tol: tolerance for convergence
        max_iter: maximum number of iterations
        k_max: maximum damping steps
    
    Returns:
        tuple: (solution, iterations, convergence_info)
    """
    x = np.copy(x0)
    convergence_info = []
    
    for i in range(max_iter):
        f_val = f_func(x)
        df_val = df_func(x)
        
        # Newton step
        try:
            delta = np.linalg.solve(df_val, -f_val)
        except np.linalg.LinAlgError:
            print(f"Linear algebra error at iteration {i}")
            break
        
        # Damping
        f_norm_current = np.linalg.norm(f_val, 2)
        
        # Find minimum k such that ||f(x + delta/2^k)||_2 < ||f(x)||_2
        k = 0
        damping_factor = 1.0
        while k <= k_max:
            x_trial = x + damping_factor * delta
            f_trial = f_func(x_trial)
            f_norm_trial = np.linalg.norm(f_trial, 2)
            
            if f_norm_trial < f_norm_current:
                break
            
            k += 1
            damping_factor = 1.0 / (2**k)
        
        if k > k_max:
            k = 0  # Use full step if no improvement found
            damping_factor = 1.0
        
        x_new = x + damping_factor * delta
        
        # Store convergence information
        convergence_info.append({
            'iteration': i,
            'x': np.copy(x),
            'f_norm': f_norm_current,
            'delta_norm': np.linalg.norm(damping_factor * delta, 2),
            'damping_factor': damping_factor
        })
        
        # Check convergence
        if f_norm_current < tol:
            print(f"Converged after {i+1} iterations")
            return x_new, i+1, convergence_info
        
        x = x_new
    
    print(f"Did not converge after {max_iter} iterations")
    return x, max_iter, convergence_info


def newton_2d_manual(f1_str, f2_str, x0, y0, max_iter=10, tol=1e-6, show_steps=True):
    """
    Newton's method for 2D systems with detailed step-by-step output
    Perfect for manual verification during exam
    """
    x, y = sp.symbols('x y')
    f1, f2 = sp.sympify(f1_str), sp.sympify(f2_str)
    
    # Calculate Jacobian symbolically
    jac = sp.Matrix([[f1.diff(x), f1.diff(y)], [f2.diff(x), f2.diff(y)]])
    
    # Convert to numerical functions
    f_func = sp.lambdify([x, y], [f1, f2], 'numpy')
    jac_func = sp.lambdify([x, y], jac, 'numpy')
    
    if show_steps:
        print(f"Newton's Method: f1 = {f1_str}, f2 = {f2_str}")
        print(f"Starting point: x⁰ = [{x0}, {y0}]")
        print("\nJacobian matrix:")
        sp.pprint(jac)
        print()
    
    x_curr, y_curr = x0, y0
    
    for k in range(max_iter):
        # Evaluate function and Jacobian
        f_val = np.array(f_func(x_curr, y_curr))
        jac_val = np.array(jac_func(x_curr, y_curr), dtype=float)
        
        if show_steps:
            print(f"Iteration {k}:")
            print(f"  x^({k}) = [{x_curr:.6f}, {y_curr:.6f}]")
            print(f"  f(x^({k})) = [{f_val[0]:.6f}, {f_val[1]:.6f}]")
            print(f"  ||f(x^({k}))||₂ = {np.linalg.norm(f_val):.6f}")
            print(f"  Df(x^({k})) = {jac_val}")
        
        # Check convergence
        if np.linalg.norm(f_val) < tol:
            if show_steps:
                print(f"  ✅ Converged after {k+1} iterations!")
            return x_curr, y_curr, k+1
        
        # Newton step
        try:
            delta = np.linalg.solve(jac_val, -f_val)
        except np.linalg.LinAlgError:
            print(f"  ❌ Singular Jacobian at iteration {k}")
            return x_curr, y_curr, k
        
        if show_steps:
            print(f"  δ^({k}) = {delta}")
            print(f"  ||δ^({k})||₂ = {np.linalg.norm(delta):.6f}")
        
        x_curr += delta[0]
        y_curr += delta[1]
        
        if show_steps:
            print(f"  x^({k+1}) = [{x_curr:.6f}, {y_curr:.6f}]")
            print()
    
    print(f"❌ No convergence after {max_iter} iterations")
    return x_curr, y_curr, max_iter


def quick_newton_2d(f1_str, f2_str, x0, y0, tol=1e-6, max_iter=20):
    """
    Quick Newton's method for 2D system using string expressions
    
    Args:
        f1_str, f2_str: string expressions for f1(x,y) and f2(x,y)
        x0, y0: initial guess
        tol: tolerance
        max_iter: maximum iterations
    
    Example:
        sol = quick_newton_2d("x**2 + y - 11", "x + y**2 - 7", 1, 1)
    """
    x, y = sp.symbols('x y')
    
    # Parse expressions
    f1 = sp.sympify(f1_str)
    f2 = sp.sympify(f2_str)
    
    # Calculate Jacobian
    jac = sp.Matrix([[f1.diff(x), f1.diff(y)],
                     [f2.diff(x), f2.diff(y)]])
    
    # Convert to numerical functions
    f_func = sp.lambdify([x, y], [f1, f2], 'numpy')
    jac_func = sp.lambdify([x, y], jac, 'numpy')
    
    # Newton iteration
    x_curr, y_curr = x0, y0
    
    for i in range(max_iter):
        f_val = np.array(f_func(x_curr, y_curr))
        jac_val = np.array(jac_func(x_curr, y_curr))
        
        # Newton step
        delta = np.linalg.solve(jac_val, -f_val)
        x_curr += delta[0]
        y_curr += delta[1]
        
        # Check convergence
        if np.linalg.norm(f_val) < tol:
            print(f"Converged in {i+1} iterations")
            return x_curr, y_curr, i+1
    
    print(f"No convergence after {max_iter} iterations")
    return x_curr, y_curr, max_iter