"""
Exam Recipes (Kochrezepte) - Step-by-step procedures for common exam problems
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# =============================================================================
# KR 1: NEWTON'S METHOD FOR NONLINEAR SYSTEMS
# =============================================================================

def kr_newton_method_manual():
    """
    KR: Newton's Method for Nonlinear Systems (Manual Calculation)
    
    Problem: Solve f(x1, x2) = [f1(x1,x2), f2(x1,x2)] = [0, 0]
    
    Steps:
    1. Set up the system f(x) = 0
    2. Calculate Jacobian matrix Df(x)
    3. Choose starting point x^(0)
    4. For each iteration k:
       a) Calculate f(x^(k))
       b) Calculate Df(x^(k))
       c) Solve: Df(x^(k)) * δ^(k) = -f(x^(k))
       d) Update: x^(k+1) = x^(k) + δ^(k)
       e) Check convergence: ||f(x^(k))|| < tolerance
    """
    
    print("KR 1: Newton's Method for Nonlinear Systems")
    print("="*50)
    
    # Example: f1 = 20 - 18*x1 - 2*x2^2, f2 = -4*x2*(x1 - x2^2)
    print("Example: Solve system")
    print("f1(x1,x2) = 20 - 18*x1 - 2*x2² = 0")
    print("f2(x1,x2) = -4*x2*(x1 - x2²) = 0")
    print()
    
    print("Step 1: Calculate Jacobian matrix")
    print("∂f1/∂x1 = -18,  ∂f1/∂x2 = -4*x2")
    print("∂f2/∂x1 = -4*x2,  ∂f2/∂x2 = -4*(x1 - 3*x2²)")
    print()
    print("Df(x1,x2) = [[-18,      -4*x2    ]")
    print("            [-4*x2, -4*(x1-3*x2²)]]")
    print()
    
    print("Step 2: Choose starting point x^(0) = [1.1, 0.9]")
    print()
    
    # First iteration
    x1, x2 = 1.1, 0.9
    print("Step 3: First iteration (k=0)")
    print(f"x^(0) = [{x1}, {x2}]")
    
    # Calculate f(x^(0))
    f1 = 20 - 18*x1 - 2*x2**2
    f2 = -4*x2*(x1 - x2**2)
    print(f"f(x^(0)) = [{f1:.6f}, {f2:.6f}]")
    print(f"||f(x^(0))||₂ = {np.sqrt(f1**2 + f2**2):.6f}")
    
    # Calculate Jacobian
    df11 = -18
    df12 = -4*x2
    df21 = -4*x2
    df22 = -4*(x1 - 3*x2**2)
    
    Df = np.array([[df11, df12], [df21, df22]])
    print(f"Df(x^(0)) = {Df}")
    
    # Solve for delta
    f_vec = np.array([f1, f2])
    delta = np.linalg.solve(Df, -f_vec)
    print(f"δ^(0) = {delta}")
    print(f"||δ^(0)||₂ = {np.linalg.norm(delta):.6f}")
    
    # Update
    x_new = np.array([x1, x2]) + delta
    print(f"x^(1) = x^(0) + δ^(0) = {x_new}")
    print()
    
    print("Continue iterations until ||f(x^(k))|| < tolerance or ||δ^(k)|| < tolerance")