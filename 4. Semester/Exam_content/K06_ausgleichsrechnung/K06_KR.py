"""
Exam Recipes (Kochrezepte) - Step-by-step procedures for common exam problems
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# =============================================================================
# KR 2: LAGRANGE INTERPOLATION
# =============================================================================

def kr_lagrange_interpolation():
    """
    KR: Lagrange Interpolation
    
    Problem: Find polynomial P(x) that passes through given points
    
    Steps:
    1. Given points (x0,y0), (x1,y1), ..., (xn,yn)
    2. Calculate Lagrange basis polynomials li(x)
    3. Construct interpolating polynomial P(x) = Σ yi * li(x)
    4. Evaluate P(x) at desired point
    """
    
    print("KR 2: Lagrange Interpolation")
    print("="*50)
    
    print("Problem: Find polynomial through points and evaluate at specific x")
    print()
    print("Step 1: For n+1 points (x0,y0), ..., (xn,yn)")
    print("Step 2: Calculate Lagrange basis polynomials:")
    print("        li(x) = ∏(j≠i) (x-xj)/(xi-xj)")
    print("Step 3: P(x) = Σ(i=0 to n) yi * li(x)")
    print()
    
    # Example calculation
    print("Example: Points (0,1013), (2500,747), (5000,540), (10000,226)")
    print("Find pressure at height 3750m")
    print()
    
    x_data = [0, 2500, 5000, 10000]
    y_data = [1013, 747, 540, 226]
    x_eval = 3750
    
    print("Step 4: Calculate each li(3750):")
    
    for i in range(len(x_data)):
        li_val = 1.0
        li_str = f"l{i}(3750) = "
        
        for j in range(len(x_data)):
            if i != j:
                factor = (x_eval - x_data[j]) / (x_data[i] - x_data[j])
                li_val *= factor
                li_str += f"(3750-{x_data[j]})/({x_data[i]}-{x_data[j]}) * "
        
        li_str = li_str.rstrip(" * ")
        print(f"{li_str} = {li_val:.6f}")
    
    print()
    print("Step 5: P(3750) = Σ yi * li(3750)")
    
    result = 0
    for i in range(len(x_data)):
        li_val = 1.0
        for j in range(len(x_data)):
            if i != j:
                li_val *= (x_eval - x_data[j]) / (x_data[i] - x_data[j])
        
        contribution = y_data[i] * li_val
        result += contribution
        print(f"        + {y_data[i]} * {li_val:.6f} = {contribution:.3f}")
    
    print(f"P(3750) = {result:.1f}")

# =============================================================================
# KR 3: NATURAL CUBIC SPLINE
# =============================================================================

def kr_natural_cubic_spline():
    """
    KR: Natural Cubic Spline Construction
    
    Steps:
    1. Given points (x0,y0), ..., (xn,yn)
    2. Set ai = yi
    3. Calculate hi = xi+1 - xi
    4. Set up tridiagonal system for ci coefficients
    5. Solve for c1, ..., cn-1 (with c0 = cn = 0)
    6. Calculate bi and di coefficients
    7. Write spline functions Si(x) for each interval
    """
    
    print("KR 3: Natural Cubic Spline")
    print("="*50)
    
    print("Problem: Construct natural cubic spline through given points")
    print()
    print("Given: Points (x0,y0), (x1,y1), ..., (xn,yn)")
    print("Goal: Find Si(x) = ai + bi(x-xi) + ci(x-xi)² + di(x-xi)³")
    print("      for each interval [xi, xi+1]")
    print()
    
    # Example with 4 points
    x_data = [4, 6, 8, 10]
    y_data = [6, 3, 9, 0]
    print(f"Example: Points {list(zip(x_data, y_data))}")
    print()
    
    n = len(x_data) - 1  # number of intervals
    
    print("Step 1: ai = yi")
    a = y_data.copy()
    for i in range(len(a)):
        print(f"a{i} = {a[i]}")
    print()
    
    print("Step 2: hi = xi+1 - xi")
    h = []
    for i in range(n):
        hi = x_data[i+1] - x_data[i]
        h.append(hi)
        print(f"h{i} = {x_data[i+1]} - {x_data[i]} = {hi}")
    print()
    
    print("Step 3: Set up tridiagonal system for c coefficients")
    print("Natural spline: c0 = cn = 0")
    print("System: A*c = b where c = [c1, c2, ..., cn-1]")
    print()
    
    # Build tridiagonal system (simplified for example)
    if n >= 3:
        print("For n=3 intervals, system is:")
        print("2(h0+h1)*c1 + h1*c2 = 3[(y2-y1)/h1 - (y1-y0)/h0]")
        print("h1*c1 + 2(h1+h2)*c2 = 3[(y3-y2)/h2 - (y2-y1)/h1]")
        print()
        
        # Calculate right-hand side
        rhs1 = 3 * ((y_data[2]-y_data[1])/h[1] - (y_data[1]-y_data[0])/h[0])
        rhs2 = 3 * ((y_data[3]-y_data[2])/h[2] - (y_data[2]-y_data[1])/h[1])
        
        print(f"RHS1 = 3[({y_data[2]}-{y_data[1]})/{h[1]} - ({y_data[1]}-{y_data[0]})/{h[0]}] = {rhs1:.4f}")
        print(f"RHS2 = 3[({y_data[3]}-{y_data[2]})/{h[2]} - ({y_data[2]}-{y_data[1]})/{h[1]}] = {rhs2:.4f}")
    
    print()
    print("Step 4: Solve system for c coefficients")
    print("Step 5: Calculate b and d coefficients:")
    print("        bi = (yi+1-yi)/hi - hi(ci+1+2ci)/3")
    print("        di = (ci+1-ci)/(3hi)")
    print()
    print("Step 6: Write final spline functions Si(x)")

# =============================================================================
# KR 4: LINEAR LEAST SQUARES
# =============================================================================

def kr_linear_least_squares():
    """
    KR: Linear Least Squares Fitting
    
    Steps:
    1. Choose basis functions f1(x), f2(x), ..., fm(x)
    2. Set up design matrix A
    3. Form normal equations A^T A λ = A^T y
    4. Solve using QR decomposition (preferred) or direct inversion
    5. Calculate residual
    """
    
    print("KR 4: Linear Least Squares")
    print("="*50)
    
    print("Problem: Fit f(x) = λ1*f1(x) + λ2*f2(x) + ... + λm*fm(x)")
    print("to data points (x1,y1), ..., (xn,yn)")
    print()
    
    print("Step 1: Choose basis functions")
    print("Example: Quadratic polynomial f(x) = λ1 + λ2*x + λ3*x²")
    print("Basis functions: f1(x) = 1, f2(x) = x, f3(x) = x²")
    print()
    
    print("Step 2: Build design matrix A")
    print("A[i,j] = fj(xi)")
    print("A = [[f1(x1), f2(x1), f3(x1)]")
    print("     [f1(x2), f2(x2), f3(x2)]")
    print("     [  ...  ,   ...  ,   ... ]")
    print("     [f1(xn), f2(xn), f3(xn)]]")
    print()
    
    print("Step 3: Set up normal equations")
    print("A^T A λ = A^T y")
    print()
    
    print("Step 4: Solve system")
    print("Method 1 (Direct): λ = (A^T A)^(-1) A^T y")
    print("Method 2 (QR): A = QR, then R λ = Q^T y")
    print("QR method is numerically more stable!")
    print()
    
    print("Step 5: Calculate residual")
    print("Residual = ||y - A λ||²")
    print()
    
    # Example calculation
    print("Example calculation steps:")
    print("1. Calculate A^T A and A^T y")
    print("2. Solve linear system")
    print("3. Coefficients give fitted polynomial")
    print("4. Evaluate at any x: f(x) = λ1 + λ2*x + λ3*x²")

# =============================================================================
# KR 5: GAUSS-NEWTON METHOD
# =============================================================================

def kr_gauss_newton():
    """
    KR: Gauss-Newton Method for Nonlinear Least Squares
    
    Steps:
    1. Define model function f(x, λ) and data (xi, yi)
    2. Set up residual function g(λ) = y - f(x, λ)
    3. Calculate Jacobian Dg(λ)
    4. Iteratively solve: Dg^T Dg δ = Dg^T g
    5. Update: λ^(k+1) = λ^(k) + δ^(k)
    6. Optional: Add damping for stability
    """
    
    print("KR 5: Gauss-Newton Method")
    print("="*50)
    
    print("Problem: Fit nonlinear model f(x, λ) to data")
    print("Example: f(x, λ) = λ1 * exp(λ2 * x)")
    print()
    
    print("Step 1: Define residual function")
    print("gi(λ) = yi - f(xi, λ) for each data point i")
    print("g(λ) = [g1(λ), g2(λ), ..., gn(λ)]^T")
    print()
    
    print("Step 2: Calculate Jacobian of residuals")
    print("For f(x, λ) = λ1 * exp(λ2 * x):")
    print("∂f/∂λ1 = exp(λ2 * x)")
    print("∂f/∂λ2 = λ1 * x * exp(λ2 * x)")
    print()
    print("Dg[i,j] = -∂f/∂λj (xi, λ)")
    print()
    
    print("Step 3: Gauss-Newton iteration")
    print("For k = 0, 1, 2, ...")
    print("a) Calculate g(λ^(k)) and Dg(λ^(k))")
    print("b) Solve: Dg^T Dg δ^(k) = Dg^T g(λ^(k))")
    print("c) Update: λ^(k+1) = λ^(k) + δ^(k)")
    print("d) Check convergence: ||g(λ^(k))|| < tolerance")
    print()
    
    print("Step 4: Damping (optional)")
    print("If ||g(λ^(k) + δ^(k))|| ≥ ||g(λ^(k))||:")
    print("Try λ^(k+1) = λ^(k) + δ^(k)/2^p for p = 1,2,3,...")
    print("until error decreases")