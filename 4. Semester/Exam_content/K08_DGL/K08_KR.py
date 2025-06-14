"""
Exam Recipes (Kochrezepte) - Step-by-step procedures for common exam problems
ODE = DGL (Differentialgleichungen)
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# =============================================================================
# KR 8: ODE SOLUTION METHODS
# =============================================================================

def kr_ode_methods():
    """
    KR: ODE Solution Methods
    
    For y' = f(x,y), y(x0) = y0, step size h
    
    Methods:
    1. Euler: y[i+1] = y[i] + h*f(x[i], y[i])
    2. Midpoint: Use f at midpoint
    3. Modified Euler: Average of two slopes
    4. Runge-Kutta 4: Weighted average of four slopes
    """
    
    print("KR 8: ODE Solution Methods")
    print("="*50)
    
    print("Problem: Solve y' = f(x,y), y(x0) = y0")
    print("Step size: h, Number of steps: n")
    print("Goal: Find y(x0 + n*h)")
    print()
    
    print("EULER METHOD:")
    print("y[i+1] = y[i] + h * f(x[i], y[i])")
    print("x[i+1] = x[i] + h")
    print("Order: 1 (error ~ h)")
    print()
    
    print("MIDPOINT METHOD:")
    print("x_half = x[i] + h/2")
    print("y_half = y[i] + (h/2) * f(x[i], y[i])")
    print("y[i+1] = y[i] + h * f(x_half, y_half)")
    print("Order: 2 (error ~ h²)")
    print()
    
    print("MODIFIED EULER (HEUN'S METHOD):")
    print("k1 = f(x[i], y[i])")
    print("y_euler = y[i] + h * k1")
    print("k2 = f(x[i] + h, y_euler)")
    print("y[i+1] = y[i] + h * (k1 + k2)/2")
    print("Order: 2 (error ~ h²)")
    print()
    
    print("RUNGE-KUTTA 4:")
    print("k1 = f(x[i], y[i])")
    print("k2 = f(x[i] + h/2, y[i] + h*k1/2)")
    print("k3 = f(x[i] + h/2, y[i] + h*k2/2)")
    print("k4 = f(x[i] + h, y[i] + h*k3)")
    print("y[i+1] = y[i] + h*(k1 + 2*k2 + 2*k3 + k4)/6")
    print("Order: 4 (error ~ h⁴)")
    print()
    
    print("Example: y' = x² + 0.1*y, y(0) = 2, h = 0.2")
    print("First step with Euler:")
    print("y[1] = y[0] + h * f(x[0], y[0])")
    print("     = 2 + 0.2 * (0² + 0.1*2)")
    print("     = 2 + 0.2 * 0.2 = 2.04")

# =============================================================================
# KR 9: HIGHER-ORDER ODE TO SYSTEM
# =============================================================================

def kr_higher_order_to_system():
    """
    KR: Converting Higher-Order ODE to First-Order System
    
    Steps:
    1. Write ODE in standard form: y^(n) = f(x, y, y', ..., y^(n-1))
    2. Introduce new variables: z1=y, z2=y', ..., zn=y^(n-1)
    3. Write system: z1'=z2, z2'=z3, ..., zn'=f(x,z1,z2,...,zn)
    4. Set initial conditions for all zi
    """
    
    print("KR 9: Higher-Order ODE to System")
    print("="*50)
    
    print("Problem: Convert y'' + ay' + by = g(x) to first-order system")
    print()
    
    print("Step 1: Solve for highest derivative")
    print("y'' = g(x) - ay' - by")
    print()
    
    print("Step 2: Introduce new variables")
    print("z1 = y")
    print("z2 = y'")
    print()
    
    print("Step 3: Write first-order system")
    print("z1' = z2")
    print("z2' = y'' = g(x) - a*z2 - b*z1")
    print()
    
    print("Step 4: Vector form")
    print("z' = [z1']  = [    z2    ]")
    print("     [z2']    [g(x)-a*z2-b*z1]")
    print()
    
    print("Step 5: Initial conditions")
    print("z1(x0) = y(x0)")
    print("z2(x0) = y'(x0)")
    print()
    
    print("Example: y'' + 0.5*y' + 2*y = sin(x)")
    print("y(0) = 1, y'(0) = 0")
    print()
    print("System:")
    print("z1' = z2")
    print("z2' = sin(x) - 0.5*z2 - 2*z1")
    print("z1(0) = 1, z2(0) = 0")
    print()
    print("Solve using any ODE method (Euler, RK4, etc.)")
    print("Solution: y(x) = z1(x)")

# =============================================================================
# KR 10: ERROR ANALYSIS
# =============================================================================

def kr_error_analysis():
    """
    KR: Error Analysis for Numerical Methods
    
    Types of errors:
    1. Absolute error: |exact - approximate|
    2. Relative error: |exact - approximate| / |exact|
    3. Convergence order: error ~ h^p
    """
    
    print("KR 10: Error Analysis")
    print("="*50)
    
    print("Problem: Analyze accuracy of numerical methods")
    print()
    
    print("Step 1: Calculate errors")
    print("Absolute error = |y_exact - y_numerical|")
    print("Relative error = |y_exact - y_numerical| / |y_exact|")
    print()
    
    print("Step 2: Check convergence order")
    print("For method with order p:")
    print("error ≈ C * h^p")
    print("log(error) ≈ log(C) + p * log(h)")
    print()
    
    print("Step 3: Plot log(error) vs log(h)")
    print("Slope = convergence order p")
    print()
    
    print("Theoretical orders:")
    print("• Euler: p = 1")
    print("• Midpoint/Modified Euler: p = 2") 
    print("• Runge-Kutta 4: p = 4")
    print("• Trapezoidal rule: p = 2")
    print("• Simpson's rule: p = 4")
    print()
    
    print("Step 4: Verify error bounds")
    print("Rectangle: |error| ≤ h²(b-a)|f''|_max / 24")
    print("Trapezoidal: |error| ≤ h²(b-a)|f''|_max / 12")
    print("Simpson: |error| ≤ h⁴(b-a)|f''''|_max / 2880")

