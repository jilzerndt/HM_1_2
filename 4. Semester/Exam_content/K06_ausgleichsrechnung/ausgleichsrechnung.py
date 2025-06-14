import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
from scipy.linalg import qr, solve
import sympy as sp
from support_functions.plotting import plot_data_and_fit

# =============================================================================
# LINEAR LEAST SQUARES
# =============================================================================

def linear_least_squares(x_data, y_data, basis_functions):
    """
    Linear least squares fitting
    
    Args:
        x_data: x-coordinates of data points
        y_data: y-coordinates of data points
        basis_functions: list of basis functions [f1, f2, ..., fm]
    
    Returns:
        coefficients: fitted coefficients
        A: design matrix
        residual: residual sum of squares
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data)
    m = len(basis_functions)
    
    # Build design matrix A
    A = np.zeros((n, m))
    for i, x in enumerate(x_data):
        for j, func in enumerate(basis_functions):
            A[i, j] = func(x)
    
    # Solve normal equations: A^T A lambda = A^T y
    # Using QR decomposition for better numerical stability
    Q, R = np.linalg.qr(A)
    coefficients = np.linalg.solve(R, Q.T @ y_data)
    
    # Calculate residual
    y_fitted = A @ coefficients
    residual = np.sum((y_data - y_fitted)**2)
    
    return coefficients, A, residual

def quick_linear_fit(x_data, y_data, basis_functions_str):
    """
    Quick linear least squares fit
    
    Args:
        x_data, y_data: data points
        basis_functions_str: list of string expressions for basis functions
    
    Example:
        coeffs = quick_linear_fit(x, y, ["1", "x", "x**2"])  # polynomial
        coeffs = quick_linear_fit(x, y, ["1", "x", "sin(x)"])  # custom basis
    """
    x = sp.Symbol('x')
    basis_funcs = [sp.lambdify(x, sp.sympify(expr), 'numpy') for expr in basis_functions_str]
    
    # Build design matrix
    A = np.zeros((len(x_data), len(basis_funcs)))
    for i, x_val in enumerate(x_data):
        for j, func in enumerate(basis_funcs):
            A[i, j] = func(x_val)
    
    # Solve normal equations using QR decomposition
    Q, R = np.linalg.qr(A)
    coeffs = np.linalg.solve(R, Q.T @ y_data)
    
    print(f"Fitted coefficients: {coeffs}")
    
    # Calculate residual
    y_fit = A @ coeffs
    residual = np.sum((y_data - y_fit)**2)
    print(f"Residual sum of squares: {residual:.6f}")
    
    return coeffs, residual


def linear_regression_manual(x_data, y_data, degree=1, use_qr=True, show_steps=True):
    """
    Linear regression with manual normal equations setup
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(x_data)
    
    if show_steps:
        print(f"Linear Regression: Polynomial degree {degree}")
        print("="*40)
        print(f"Data points: {n}")
        print("Building design matrix A...")
    
    # Build design matrix
    A = np.zeros((n, degree + 1))
    for i in range(n):
        for j in range(degree + 1):
            A[i, j] = x_data[i] ** j
    
    if show_steps:
        print(f"Design matrix A ({n} × {degree+1}):")
        print(A)
        print()
    
    if use_qr:
        if show_steps:
            print("Method: QR decomposition")
        Q, R = qr(A)
        coeffs = solve(R, Q.T @ y_data)
        
        if show_steps:
            print(f"Condition number of R: {np.linalg.cond(R):.2e}")
    else:
        if show_steps:
            print("Method: Normal equations")
        AtA = A.T @ A
        Atb = A.T @ y_data
        coeffs = solve(AtA, Atb)
        
        if show_steps:
            print("A^T A:")
            print(AtA)
            print(f"A^T b: {Atb}")
            print(f"Condition number of A^T A: {np.linalg.cond(AtA):.2e}")
    
    # Calculate residual
    y_fitted = A @ coeffs
    residual = np.sum((y_data - y_fitted)**2)
    
    if show_steps:
        print(f"Coefficients: {coeffs}")
        print(f"Residual sum of squares: {residual:.6f}")
        
        # Write polynomial
        poly_str = " + ".join([f"{coeffs[i]:.4f}x^{i}" if i > 0 else f"{coeffs[i]:.4f}" 
                              for i in range(len(coeffs))])
        print(f"Fitted polynomial: y = {poly_str}")
    
    return coeffs, residual, A

# =============================================================================
# NONLINEAR LEAST SQUARES
# =============================================================================

def gauss_newton_method(x_data, y_data, model_func, jacobian_func, initial_params, 
                       max_iter=50, tol=1e-6, damped=True):
    """
    Gauss-Newton method for nonlinear least squares
    
    Args:
        x_data: x-coordinates of data points
        y_data: y-coordinates of data points
        model_func: model function f(x, params)
        jacobian_func: Jacobian function df/dparams(x, params)
        initial_params: initial parameter guess
        max_iter: maximum iterations
        tol: convergence tolerance
        damped: use damping if True
    
    Returns:
        params: fitted parameters
        iterations: number of iterations
        residual_history: history of residuals
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    params = np.array(initial_params, dtype=float)
    
    residual_history = []
    
    for iteration in range(max_iter):
        # Calculate residuals
        y_model = np.array([model_func(x, params) for x in x_data])
        residuals = y_data - y_model
        
        # Calculate Jacobian matrix
        J = np.zeros((len(x_data), len(params)))
        for i, x in enumerate(x_data):
            J[i, :] = jacobian_func(x, params)
        
        # Gauss-Newton step: solve J^T J delta = J^T residuals
        try:
            JTJ = J.T @ J
            JTr = J.T @ residuals
            delta = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            print(f"Singular matrix at iteration {iteration}")
            break
        
        # Current error
        current_error = np.linalg.norm(residuals)**2
        residual_history.append(current_error)
        
        # Damping if requested
        if damped:
            damping_factor = 1.0
            for k in range(4):  # Try up to 4 damping steps
                params_trial = params + damping_factor * delta
                y_trial = np.array([model_func(x, params_trial) for x in x_data])
                residuals_trial = y_data - y_trial
                trial_error = np.linalg.norm(residuals_trial)**2
                
                if trial_error < current_error:
                    break
                damping_factor *= 0.5
            
            params = params + damping_factor * delta
        else:
            params = params + delta
        
        # Check convergence
        if np.linalg.norm(delta) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return params, iteration + 1, residual_history

# =============================================================================
# POLYNOMIAL LEAST SQUARES
# =============================================================================

def polynomial_least_squares(x_data, y_data, degree):
    """
    Polynomial least squares fitting
    
    Args:
        x_data: x-coordinates of data points
        y_data: y-coordinates of data points
        degree: degree of polynomial
    
    Returns:
        coefficients: polynomial coefficients [a0, a1, ..., an]
        residual: residual sum of squares
    """
    # Create basis functions for polynomial
    basis_functions = [lambda x, k=k: x**k for k in range(degree + 1)]
    
    coefficients, A, residual = linear_least_squares(x_data, y_data, basis_functions)
    
    return coefficients, residual

def quick_polyfit(x_data, y_data, degree, plot=True):
    """Quick polynomial fitting"""
    coeffs = np.polyfit(x_data, y_data, degree)
    
    if plot:
        x_plot = np.linspace(min(x_data), max(x_data), 100)
        y_plot = np.polyval(coeffs, x_plot)
        plot_data_and_fit(x_data, y_data, x_plot, y_plot, 
                         f"Polynomial Fit (degree {degree})")
    
    print(f"Polynomial coefficients (highest to lowest degree): {coeffs}")
    return coeffs