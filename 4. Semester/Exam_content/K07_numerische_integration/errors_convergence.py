import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sympy as sp
import math
from int_rules import rectangle_rule, trapezoidal_rule, simpson_rule


def error_analysis(f, f_second_deriv, f_fourth_deriv, a, b, h):
    """
    Theoretical error analysis for integration rules
    
    Args:
        f: function to integrate
        f_second_deriv: second derivative of f
        f_fourth_deriv: fourth derivative of f
        a, b: integration limits
        h: step size
    
    Returns:
        dict with error bounds for different methods
    """
    # Find maximum of second and fourth derivatives on [a, b]
    x_test = np.linspace(a, b, 1000)
    
    try:
        max_f2 = np.max(np.abs(f_second_deriv(x_test)))
        max_f4 = np.max(np.abs(f_fourth_deriv(x_test)))
    except:
        # If derivatives can't be evaluated, return None
        return None
    
    errors = {
        'rectangle': h**2 / 24 * (b - a) * max_f2,
        'trapezoidal': h**2 / 12 * (b - a) * max_f2,
        'simpson': h**4 / 2880 * (b - a) * max_f4
    }
    
    return errors


def convergence_study(f, exact_value, a, b, methods):
    """
    Study convergence of different integration methods
    
    Args:
        f: function to integrate
        exact_value: exact value of integral
        a, b: integration limits
        methods: list of method names to test
    
    Returns:
        dict with convergence data
    """
    n_values = [2**i for i in range(1, 11)]  # n = 2, 4, 8, ..., 1024
    results = {method: [] for method in methods}
    
    for n in n_values:
        h = (b - a) / n
        
        if 'rectangle' in methods:
            approx = rectangle_rule(f, a, b, n)
            error = abs(exact_value - approx)
            results['rectangle'].append((h, error))
        
        if 'trapezoidal' in methods:
            approx = trapezoidal_rule(f, a, b, n)
            error = abs(exact_value - approx)
            results['trapezoidal'].append((h, error))
        
        if 'simpson' in methods and n % 2 == 0:
            approx = simpson_rule(f, a, b, n)
            error = abs(exact_value - approx)
            results['simpson'].append((h, error))
    
    return results


def integration_comparison(f_str, a, b, n=4, show_all_steps=True):
    """
    Compare all integration methods with detailed calculations
    """
    x_sym = sp.Symbol('x')
    f_expr = sp.sympify(f_str)
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    
    print(f"Numerical Integration Comparison")
    print("="*50)
    print(f"Function: f(x) = {f_str}")
    print(f"Interval: [{a}, {b}]")
    print(f"Number of intervals: n = {n}")
    
    h = (b - a) / n
    print(f"Step size: h = {h}")
    print()
    
    # Get exact value if possible
    try:
        exact_expr = sp.integrate(f_expr, (x_sym, a, b))
        exact_value = float(exact_expr)
        print(f"Exact value: {exact_value:.8f}")
    except:
        exact_value = None
        print("Exact value: Not available")
    print()
    
    results = {}
    
    # Rectangle Rule (Midpoint)
    if show_all_steps:
        print("1. RECTANGLE RULE (Midpoint)")
        print("-" * 30)
    x_mid = np.linspace(a + h/2, b - h/2, n)
    y_mid = f_func(x_mid)
    rect_result = h * np.sum(y_mid)
    
    if show_all_steps:
        print(f"Midpoints: {x_mid}")
        print(f"Function values: {y_mid}")
        print(f"R = h × Σf(xi) = {h} × {np.sum(y_mid):.6f} = {rect_result:.8f}")
        if exact_value:
            print(f"Error: {abs(exact_value - rect_result):.2e}")
        print()
    
    results['rectangle'] = rect_result
    
    # Trapezoidal Rule
    if show_all_steps:
        print("2. TRAPEZOIDAL RULE")
        print("-" * 20)
    x_trap = np.linspace(a, b, n + 1)
    y_trap = f_func(x_trap)
    trap_result = h * (0.5 * (y_trap[0] + y_trap[-1]) + np.sum(y_trap[1:-1]))
    
    if show_all_steps:
        print(f"Grid points: {x_trap}")
        print(f"Function values: {y_trap}")
        print(f"T = h × [(f(a) + f(b))/2 + Σf(xi)]")
        print(f"  = {h} × [({y_trap[0]} + {y_trap[-1]})/2 + {np.sum(y_trap[1:-1]):.6f}]")
        print(f"  = {trap_result:.8f}")
        if exact_value:
            print(f"Error: {abs(exact_value - trap_result):.2e}")
        print()
    
    results['trapezoidal'] = trap_result
    
    # Simpson's Rule
    if n % 2 == 0:
        if show_all_steps:
            print("3. SIMPSON'S RULE")
            print("-" * 15)
        x_simp = np.linspace(a, b, n + 1)
        y_simp = f_func(x_simp)
        simp_result = h/3 * (y_simp[0] + y_simp[-1] + 4*np.sum(y_simp[1::2]) + 2*np.sum(y_simp[2:-1:2]))
        
        if show_all_steps:
            print(f"S = h/3 × [f(a) + 4×Σf(x_odd) + 2×Σf(x_even) + f(b)]")
            print(f"Coefficients pattern: 1, 4, 2, 4, 2, ..., 4, 1")
            print(f"S = {simp_result:.8f}")
            if exact_value:
                print(f"Error: {abs(exact_value - simp_result):.2e}")
            print()
        
        results['simpson'] = simp_result
    else:
        print("Simpson's rule requires even number of intervals")
    
    return results