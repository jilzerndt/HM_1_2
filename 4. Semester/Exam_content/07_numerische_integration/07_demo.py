#!/usr/bin/env python3
"""
Chapter 7: Numerical Integration
Rectangle Rule, Trapezoidal Rule, Simpson's Rule, Romberg Extrapolation, Gauss Quadrature
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math

# Import custom integration rules and Romberg extrapolation
from int_rules import rectangle_rule, trapezoidal_rule, simpson_rule, adaptive_simpson
from romberg import romberg_extrapolation
from gauss_quad import gauss_legendre_quadrature
from errors_convergence import error_analysis, convergence_study

# Example functions for testing
def example_function_1(x):
    """f(x) = 1/x"""
    return 1/x

def example_function_2(x):
    """f(x) = cos(x^2)"""
    return np.cos(x**2)

def example_function_3(x):
    """f(x) = exp(-x^2)"""
    return np.exp(-x**2)

def example_function_1_derivatives():
    """Derivatives of f(x) = 1/x"""
    def f2(x):
        return 2 / x**3
    def f4(x):
        return 24 / x**5
    return f2, f4

# Demo functions
def demo_basic_integration():
    """Demo basic integration rules"""
    print("Basic Integration Rules Demo")
    print("-" * 40)
    
    # Test function: f(x) = 1/x from 2 to 4
    f = example_function_1
    a, b = 2, 4
    exact = np.log(4) - np.log(2)  # ln(4) - ln(2) = ln(2)
    
    print(f"Function: f(x) = 1/x")
    print(f"Interval: [{a}, {b}]")
    print(f"Exact value: {exact:.6f}")
    print()
    
    n_values = [4, 8, 16, 32]
    
    for n in n_values:
        rect = rectangle_rule(f, a, b, n)
        trap = trapezoidal_rule(f, a, b, n)
        simp = simpson_rule(f, a, b, n)
        
        print(f"n = {n:2d}:")
        print(f"  Rectangle:   {rect:.6f} (error: {abs(exact-rect):.2e})")
        print(f"  Trapezoidal: {trap:.6f} (error: {abs(exact-trap):.2e})")
        print(f"  Simpson:     {simp:.6f} (error: {abs(exact-simp):.2e})")
        print()

def demo_romberg():
    """Demo Romberg extrapolation"""
    print("Romberg Extrapolation Demo")
    print("-" * 40)
    
    # Test function: cos(x^2) from 0 to π
    f = example_function_2
    a, b = 0, np.pi
    
    print(f"Function: f(x) = cos(x²)")
    print(f"Interval: [{a}, {b}]")
    
    # Romberg extrapolation
    T, best_approx = romberg_extrapolation(f, a, b, 4)
    
    print("\nRomberg Table:")
    print("j\\k", end="")
    for k in range(5):
        print(f"        T[j,{k}]", end="")
    print()
    
    for j in range(5):
        print(f"{j:2d} ", end="")
        for k in range(5-j):
            print(f"{T[j,k]:12.8f}", end="")
        print()
    
    print(f"\nBest approximation: {best_approx:.8f}")

def demo_gauss_quadrature():
    """Demo Gauss quadrature"""
    print("\nGauss Quadrature Demo")
    print("-" * 40)
    
    # Test function: exp(-x^2) from 0 to 0.5
    f = example_function_3
    a, b = 0, 0.5
    
    print(f"Function: f(x) = exp(-x²)")
    print(f"Interval: [{a}, {b}]")
    
    # Compare with exact value (using scipy)
    exact, _ = integrate.quad(f, a, b)
    print(f"Reference value: {exact:.8f}")
    print()
    
    for n in [1, 2, 3]:
        gauss_result = gauss_legendre_quadrature(f, a, b, n)
        error = abs(exact - gauss_result)
        print(f"Gauss-{n}: {gauss_result:.8f} (error: {error:.2e})")

def demo_convergence():
    """Demo convergence analysis"""
    print("\nConvergence Analysis Demo")
    print("-" * 40)
    
    # Test function: 1/x from 2 to 4
    f = example_function_1
    a, b = 2, 4
    exact = np.log(2)
    
    methods = ['rectangle', 'trapezoidal', 'simpson']
    results = convergence_study(f, exact, a, b, methods)
    
    plt.figure(figsize=(10, 6))
    
    colors = {'rectangle': 'red', 'trapezoidal': 'blue', 'simpson': 'green'}
    
    for method in methods:
        if results[method]:
            h_values, errors = zip(*results[method])
            plt.loglog(h_values, errors, 'o-', color=colors[method], 
                      label=f'{method.capitalize()}')
    
    plt.xlabel('Step size h')
    plt.ylabel('Absolute error')
    plt.title('Convergence of Integration Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def demo_error_estimates():
    """Demo theoretical error estimates"""
    print("\nError Estimates Demo")
    print("-" * 40)
    
    # Test function: 1/x from 2 to 4
    f = example_function_1
    f2, f4 = example_function_1_derivatives()
    a, b = 2, 4
    
    n = 8
    h = (b - a) / n
    
    # Calculate actual errors
    exact = np.log(2)
    rect_approx = rectangle_rule(f, a, b, n)
    trap_approx = trapezoidal_rule(f, a, b, n)
    simp_approx = simpson_rule(f, a, b, n)
    
    actual_errors = {
        'rectangle': abs(exact - rect_approx),
        'trapezoidal': abs(exact - trap_approx),
        'simpson': abs(exact - simp_approx)
    }
    
    # Calculate theoretical error bounds
    theoretical_errors = error_analysis(f, f2, f4, a, b, h)
    
    print(f"Step size h = {h:.4f}, n = {n}")
    print()
    print("Method        Actual Error    Theoretical Bound")
    print("-" * 50)
    
    for method in ['rectangle', 'trapezoidal', 'simpson']:
        actual = actual_errors[method]
        theoretical = theoretical_errors[method] if theoretical_errors else float('nan')
        print(f"{method:12s}  {actual:.2e}      {theoretical:.2e}")

# Main execution
if __name__ == "__main__":
    print("Chapter 7: Numerical Integration")
    print("="*50)
    
    # Run all demos
    demo_basic_integration()
    demo_romberg()
    demo_gauss_quadrature()
    demo_convergence()
    demo_error_estimates()
    
    print("\nAdditional utility functions available:")
    print("- trapezoidal_rule_non_equidistant()")
    print("- adaptive_simpson()")
    print("- convergence_study()")
