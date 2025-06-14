"""
Chapter 6: Interpolation and Regression
Lagrange Interpolation, Cubic Splines, Linear/Nonlinear Least Squares, Gauss-Newton
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
import sympy as sp

# Import custom modules - for exam obviously copy and paste!!
from interpolation import lagrange_interpolation, natural_cubic_spline_coefficients, evaluate_cubic_spline
from ausgleichsrechnung import polynomial_least_squares, gauss_newton_method


def exponential_model(x, params):
    """
    Exponential model: f(x) = a * exp(b * x)
    """
    a, b = params
    return a * np.exp(b * x)

def exponential_jacobian(x, params):
    """
    Jacobian of exponential model
    """
    a, b = params
    exp_bx = np.exp(b * x)
    return np.array([exp_bx, a * x * exp_bx])

# Example functions and test cases
def demo_lagrange():
    """Demo Lagrange interpolation"""
    print("Lagrange Interpolation Demo")
    print("-" * 30)
    
    # Example data
    x_data = [0, 1, 2, 3]
    y_data = [1, 2, 0, 3]
    
    # Interpolate at new points
    x_new = np.linspace(0, 3, 50)
    y_new = [lagrange_interpolation(x_data, y_data, x) for x in x_new]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ro', label='Data points', markersize=8)
    plt.plot(x_new, y_new, 'b-', label='Lagrange interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lagrange Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Test specific point
    test_x = 1.5
    test_y = lagrange_interpolation(x_data, y_data, test_x)
    print(f"Interpolated value at x = {test_x}: y = {test_y}")

def demo_cubic_spline():
    """Demo cubic spline interpolation"""
    print("\nCubic Spline Demo")
    print("-" * 30)
    
    # Example data
    x_data = [4, 6, 8, 10]
    y_data = [6, 3, 9, 0]
    
    # Calculate spline coefficients
    coeffs = natural_cubic_spline_coefficients(x_data, y_data)
    
    print("Spline coefficients:")
    for i in range(len(x_data)-1):
        print(f"Interval [{x_data[i]}, {x_data[i+1]}]:")
        print(f"  S_{i}(x) = {coeffs['a'][i]:.4f} + {coeffs['b'][i]:.4f}(x-{x_data[i]}) + {coeffs['c'][i]:.4f}(x-{x_data[i]})² + {coeffs['d'][i]:.4f}(x-{x_data[i]})³")
    
    # Evaluate spline
    x_eval = np.linspace(x_data[0], x_data[-1], 100)
    y_eval = [evaluate_cubic_spline(coeffs, x) for x in x_eval]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ro', label='Data points', markersize=8)
    plt.plot(x_eval, y_eval, 'b-', label='Natural cubic spline')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Natural Cubic Spline Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()

def demo_linear_regression():
    """Demo linear least squares regression"""
    print("\nLinear Regression Demo")
    print("-" * 30)
    
    # Generate example data
    np.random.seed(42)
    x_data = np.linspace(0, 10, 20)
    y_true = 2 * x_data + 1 + 0.5 * x_data**2
    y_data = y_true + 0.5 * np.random.randn(len(x_data))
    
    # Fit quadratic polynomial
    coeffs, residual = polynomial_least_squares(x_data, y_data, 2)
    
    print(f"Fitted polynomial coefficients: {coeffs}")
    print(f"Residual sum of squares: {residual}")
    
    # Plot results
    x_plot = np.linspace(0, 10, 100)
    y_fit = coeffs[0] + coeffs[1] * x_plot + coeffs[2] * x_plot**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ro', label='Data points')
    plt.plot(x_plot, y_fit, 'b-', label=f'Fitted polynomial (degree 2)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Least Squares Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()

def demo_gauss_newton():
    """Demo Gauss-Newton for nonlinear regression"""
    print("\nGauss-Newton Demo")
    print("-" * 30)
    
    # Generate exponential data with noise
    np.random.seed(42)
    x_data = np.linspace(0, 2, 15)
    true_params = [3.0, -1.0]  # a=3, b=-1
    y_true = exponential_model(x_data, true_params)
    y_data = y_true + 0.1 * np.random.randn(len(x_data))
    
    # Fit using Gauss-Newton
    initial_params = [1.0, -0.5]
    fitted_params, iterations, residual_hist = gauss_newton_method(
        x_data, y_data, exponential_model, exponential_jacobian, 
        initial_params, damped=True
    )
    
    print(f"True parameters: {true_params}")
    print(f"Initial guess: {initial_params}")
    print(f"Fitted parameters: {fitted_params}")
    print(f"Iterations: {iterations}")
    
    # Plot results
    x_plot = np.linspace(0, 2, 100)
    y_fit = exponential_model(x_plot, fitted_params)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_data, y_data, 'ro', label='Data points')
    plt.plot(x_plot, y_fit, 'b-', label=f'Fitted: y = {fitted_params[0]:.3f} * exp({fitted_params[1]:.3f} * x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Exponential Model Fitting')

