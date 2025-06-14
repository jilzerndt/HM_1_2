#!/usr/bin/env python3
"""
Chapter 5: Numerical Solution of Nonlinear Systems
Newton's Method for Systems, Jacobian Matrix, Linearization
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Import custom modules - for exam obviously copy and paste!!
from support_functions.basics import evaluate_at_point
from support_functions.plotting import plot_system_2d
from partielle_ableitungen import jacobian_symbolic, linearize_function
from newton_nps import newton_method_systems, damped_newton_method


# Example usage and test functions
def example_system_1():
    """
    Example: f1(x1,x2) = x1^2 + x2 - 11 = 0
             f2(x1,x2) = x1 + x2^2 - 7 = 0
    """
    def f(x):
        return np.array([
            x[0]**2 + x[1] - 11,
            x[0] + x[1]**2 - 7
        ])
    
    def df(x):
        return np.array([
            [2*x[0], 1],
            [1, 2*x[1]]
        ])
    
    return f, df

def example_system_2():
    """
    Example: f1(x1,x2) = 20 - 18*x1 - 2*x2^2 = 0
             f2(x1,x2) = -4*x2*(x1 - x2^2) = 0
    """
    def f(x):
        return np.array([
            20 - 18*x[0] - 2*x[1]**2,
            -4*x[1]*(x[0] - x[1]**2)
        ])
    
    def df(x):
        return np.array([
            [-18, -4*x[1]],
            [-4*x[1], -4*(x[0] - 3*x[1]**2)]
        ])
    
    return f, df



# Test the methods
if __name__ == "__main__":
    print("Chapter 5: Nonlinear Systems")
    print("="*40)
    
    # Example 1: Simple system
    print("\nExample 1: Simple quadratic system")
    f, df = example_system_1()
    
    x0 = np.array([1.0, 1.0])
    solution, iterations, info = newton_method_systems(f, df, x0)
    
    print(f"Starting point: {x0}")
    print(f"Solution: {solution}")
    print(f"f(solution): {f(solution)}")
    print(f"Iterations: {iterations}")
    
    # Example 2: More complex system with damping
    print("\nExample 2: System requiring damping")
    f2, df2 = example_system_2()
    
    x0 = np.array([1.1, 0.9])
    solution2, iterations2, info2 = damped_newton_method(f2, df2, x0)
    
    print(f"Starting point: {x0}")
    print(f"Solution: {solution2}")
    print(f"f(solution): {f2(solution2)}")
    print(f"Iterations: {iterations2}")
    
    # Symbolic example
    print("\nSymbolic Jacobian Example:")
    x1, x2 = sp.symbols('x1 x2')
    f_sym = [x1**2 + x2 - 11, x1 + x2**2 - 7]
    vars_sym = [x1, x2]
    
    jac_sym = jacobian_symbolic(f_sym, vars_sym)
    print("Symbolic Jacobian:")
    print(jac_sym)
    
    # Linearization example
    print("\nLinearization Example:")
    point = np.array([1.0, 1.0])
    lin_func, f0, jac0 = linearize_function(f_sym, vars_sym, point)
    
    print(f"Function value at point: {f0}")
    print(f"Jacobian at point: \n{jac0}")
    
    # Test linearized function
    test_point = np.array([1.1, 1.1])
    print(f"Linearized function at {test_point}: {lin_func(test_point)}")
