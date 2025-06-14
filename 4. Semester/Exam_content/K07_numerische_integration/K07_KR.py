"""
Exam Recipes (Kochrezepte) - Step-by-step procedures for common exam problems
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# =============================================================================
# KR 6: NUMERICAL INTEGRATION
# =============================================================================

def kr_numerical_integration():
    """
    KR: Numerical Integration Methods
    
    Steps for each method:
    1. Rectangle Rule: R = h * Σf(xi + h/2)
    2. Trapezoidal Rule: T = h * [f(a)+f(b)]/2 + h * Σf(xi)
    3. Simpson's Rule: S = h/3 * [f(a) + 4*Σf(x_odd) + 2*Σf(x_even) + f(b)]
    4. Romberg Extrapolation: Improve trapezoidal estimates
    """
    
    print("KR 6: Numerical Integration")
    print("="*50)
    
    print("Problem: Approximate ∫[a to b] f(x) dx")
    print()
    
    print("Step 1: Choose method and number of intervals n")
    print("Step 2: Calculate step size h = (b-a)/n")
    print("Step 3: Evaluate function at required points")
    print()
    
    print("RECTANGLE RULE (Midpoint):")
    print("xi = a + (i-0.5)*h for i = 1,2,...,n")
    print("R = h * [f(x1) + f(x2) + ... + f(xn)]")
    print()
    
    print("TRAPEZOIDAL RULE:")
    print("xi = a + i*h for i = 0,1,...,n")
    print("T = h * [(f(a) + f(b))/2 + f(x1) + f(x2) + ... + f(xn-1)]")
    print()
    
    print("SIMPSON'S RULE (n must be even):")
    print("S = h/3 * [f(a) + 4*f(x1) + 2*f(x2) + 4*f(x3) + ... + 4*f(xn-1) + f(b)]")
    print("Pattern: 1, 4, 2, 4, 2, ..., 4, 1")
    print()
    
    print("ROMBERG EXTRAPOLATION:")
    print("T[j,0] = Trapezoidal with h = (b-a)/2^j")
    print("T[j,k] = (4^k * T[j+1,k-1] - T[j,k-1]) / (4^k - 1)")
    print()
    
    # Example calculation
    print("Example: ∫[2 to 4] 1/x dx with n=4")
    print("Exact value = ln(4) - ln(2) = ln(2) ≈ 0.693147")
    print()
    print("Trapezoidal: h = 0.5")
    print("T = 0.5 * [(1/2 + 1/4)/2 + 1/2.5 + 1/3 + 1/3.5]")
    print("  = 0.5 * [0.375 + 0.4 + 0.333... + 0.285...]")
    print("  ≈ 0.697")

# =============================================================================
# KR 7: ROMBERG EXTRAPOLATION
# =============================================================================

def kr_romberg_extrapolation():
    """
    KR: Romberg Extrapolation
    
    Steps:
    1. Calculate T[j,0] using trapezoidal rule with h_j = (b-a)/2^j
    2. Apply Richardson extrapolation: T[j,k] = (4^k * T[j+1,k-1] - T[j,k-1])/(4^k-1)
    3. Continue until desired accuracy
    """
    
    print("KR 7: Romberg Extrapolation")
    print("="*50)
    
    print("Problem: Improve trapezoidal rule estimates")
    print()
    
    print("Step 1: Calculate first column T[j,0]")
    print("For j = 0,1,2,...,m:")
    print("h_j = (b-a)/2^j")
    print("T[j,0] = Trapezoidal rule with step size h_j")
    print()
    
    print("Step 2: Extrapolation formula")
    print("T[j,k] = (4^k * T[j+1,k-1] - T[j,k-1]) / (4^k - 1)")
    print()
    
    print("Step 3: Build Romberg table")
    print("   k=0      k=1      k=2      k=3")
    print("j=0 T[0,0]  T[0,1]   T[0,2]   T[0,3]")
    print("j=1 T[1,0]  T[1,1]   T[1,2]")
    print("j=2 T[2,0]  T[2,1]")
    print("j=3 T[3,0]")
    print()
    
    print("Best estimate: T[0,m] (top-right corner)")
    print()
    
    print("Example calculation:")
    print("T[0,1] = (4*T[1,0] - T[0,0])/3")
    print("T[0,2] = (16*T[1,1] - T[0,1])/15")
    print("T[1,1] = (4*T[2,0] - T[1,0])/3")