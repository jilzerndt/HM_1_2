"""
Basic support functions for the exam content module.

"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# You might also need these:
# =============================================================================
# DATA ANALYSIS HELPERS
# =============================================================================

def load_and_preview_data(data_str, has_header=True):
    """
    Load data from string format (copy-paste from exercises)
    
    Example:
        data_str = '''
        x,y
        1,2
        2,4
        3,6
        '''
        df = load_and_preview_data(data_str)
    """
    from io import StringIO
    
    if has_header:
        df = pd.read_csv(StringIO(data_str.strip()))
    else:
        lines = data_str.strip().split('\n')
        data = [list(map(float, line.split())) for line in lines if line.strip()]
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(len(data[0]))])
    
    print("Data preview:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    return df

def quick_error_analysis(exact_values, approx_values, method_name=""):
    """Quick error analysis"""
    exact_values = np.array(exact_values)
    approx_values = np.array(approx_values)
    
    abs_errors = np.abs(exact_values - approx_values)
    rel_errors = abs_errors / np.abs(exact_values)
    rel_errors[exact_values == 0] = np.inf  # Handle division by zero
    
    print(f"Error Analysis {method_name}:")
    print(f"  Max absolute error: {np.max(abs_errors):.2e}")
    print(f"  Mean absolute error: {np.mean(abs_errors):.2e}")
    print(f"  Max relative error: {np.max(rel_errors[np.isfinite(rel_errors)]):.2e}")
    print(f"  Mean relative error: {np.mean(rel_errors[np.isfinite(rel_errors)]):.2e}")
    
    return abs_errors, rel_errors

# =============================================================================
# MATRIX AND LINEAR ALGEBRA HELPERS
# =============================================================================

def evaluate_at_point(expr, variables, point):
    """
    Evaluate symbolic expression at given point
    
    Args:
        expr: SymPy expression or matrix
        variables: list of variables
        point: numpy array of values
    
    Returns:
        Numerical evaluation
    """
    subs_dict = {var: val for var, val in zip(variables, point)}
    return np.array(expr.subs(subs_dict), dtype=float)

def quick_matrix_analysis(A):
    """Quick matrix analysis"""
    A = np.array(A)
    print(f"Matrix shape: {A.shape}")
    print(f"Determinant: {np.linalg.det(A):.6f}")
    print(f"Condition number: {np.linalg.cond(A):.2e}")
    print(f"Rank: {np.linalg.matrix_rank(A)}")
    
    if A.shape[0] == A.shape[1]:
        eigenvals = np.linalg.eigvals(A)
        print(f"Eigenvalues: {eigenvals}")
    
    return {
        'det': np.linalg.det(A),
        'cond': np.linalg.cond(A),
        'rank': np.linalg.matrix_rank(A)
    }

def solve_normal_equations(A, b, use_qr=True):
    """Solve normal equations A^T A x = A^T b"""
    A = np.array(A)
    b = np.array(b)
    
    if use_qr:
        Q, R = np.linalg.qr(A)
        x = np.linalg.solve(R, Q.T @ b)
        method = "QR decomposition"
    else:
        AtA = A.T @ A
        Atb = A.T @ b
        x = np.linalg.solve(AtA, Atb)
        method = "Normal equations"
    
    print(f"Solution using {method}: {x}")
    
    # Calculate residual
    residual = np.linalg.norm(A @ x - b)**2
    print(f"Residual sum of squares: {residual:.6f}")
    
    return x, residual