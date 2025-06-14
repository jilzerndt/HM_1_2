"""
Chapter 5: Numerical Solution of Nonlinear Systems
Newton's Method for Systems, Jacobian Matrix, Linearization
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
from newton_nps import newton_method_systems, damped_newton_method


def jacobian_symbolic(functions, variables):
    """
    Calculate Jacobian matrix symbolically using SymPy
    
    Args:
        functions: list of symbolic functions
        variables: list of symbolic variables
    
    Returns:
        Jacobian matrix as SymPy matrix
    """
    f_matrix = sp.Matrix(functions)
    var_matrix = sp.Matrix(variables)
    return f_matrix.jacobian(var_matrix)

def jacobian_calculator(expressions, variables, point=None):
    """
    Calculate Jacobian matrix with optional evaluation at a point
    """
    vars_sym = sp.symbols(variables)
    exprs_sym = [sp.sympify(expr) for expr in expressions]
    
    print("Functions:")
    for i, expr in enumerate(exprs_sym):
        print(f"  f{i+1}({', '.join(variables)}) = {expr}")
    
    # Calculate Jacobian
    jac_matrix = sp.Matrix(exprs_sym).jacobian(sp.Matrix(vars_sym))
    
    print("\nJacobian matrix:")
    sp.pprint(jac_matrix)
    
    if point is not None:
        point_dict = {var: val for var, val in zip(vars_sym, point)}
        jac_numerical = np.array([[float(jac_matrix[i, j].subs(point_dict)) 
                                 for j in range(len(variables))] 
                                for i in range(len(expressions))])
        
        print(f"\nAt point {point}:")
        print(f"Jacobian = \n{jac_numerical}")
        
        # Additional analysis
        if jac_numerical.shape[0] == jac_numerical.shape[1]:
            det = np.linalg.det(jac_numerical)
            cond = np.linalg.cond(jac_numerical)
            print(f"Determinant = {det:.6f}")
            print(f"Condition number = {cond:.2e}")
        
        return jac_matrix, jac_numerical
    
    return jac_matrix

def quick_jacobian(expressions, variables):
    """
    Quick Jacobian calculation
    
    Args:
        expressions: list of string expressions
        variables: list of variable names as strings
    
    Example:
        jac = quick_jacobian(["x**2 + y", "x*y"], ["x", "y"])
    """
    vars_sym = sp.symbols(variables)
    exprs_sym = [sp.sympify(expr) for expr in expressions]
    
    jac_matrix = sp.Matrix(exprs_sym).jacobian(sp.Matrix(vars_sym))
    
    print("Symbolic Jacobian:")
    sp.pprint(jac_matrix)
    
    return jac_matrix


def linearize_function(f_symbolic, variables, x0_point):
    """
    Linearize function f around point x0
    
    Args:
        f_symbolic: SymPy function or list of functions
        variables: list of SymPy variables
        x0_point: point of linearization as numpy array
    
    Returns:
        linearized function g(x) = f(x0) + Df(x0) * (x - x0)
    """
    # Calculate function value at x0
    f0 = evaluate_at_point(f_symbolic, variables, x0_point)
    
    # Calculate Jacobian at x0
    if isinstance(f_symbolic, list):
        jac = jacobian_symbolic(f_symbolic, variables)
    else:
        jac = sp.Matrix([f_symbolic]).jacobian(sp.Matrix(variables))
    
    jac0 = evaluate_at_point(jac, variables, x0_point)
    
    def linearized_func(x):
        return f0 + jac0 @ (x - x0_point)
    
    return linearized_func, f0, jac0