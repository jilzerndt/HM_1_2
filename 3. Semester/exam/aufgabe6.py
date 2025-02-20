import numpy as np
import matplotlib.pyplot as plt
import math

"""Initialisierungswerte für a)"""

c = 5


# Gegebene Matrix A
A = np.array[[4, -13], [c, 30]]

# Startvektor
x = np.array([1, 0, 0], dtype=float)

# Toleranz und maximale Iterationen
tolerance = 1e-4
max_iterations = 1000


# von-Mises-Iteration
def von_mises_iteration(A, x, tolerance, max_iterations):
    x = x / np.linalg.norm(x)  # Normieren des Startvektors
    for k in range(max_iterations):
        x_next = np.dot(A, x)  # Matrix-Vektor-Multiplikation
        x_next = x_next / np.linalg.norm(x_next)  # Normieren des neuen Vektors

        # Abbruchbedingung
        if np.linalg.norm(x_next - x) < tolerance:
            break

        x = x_next

    # Berechnung des zugehörigen Eigenwerts
    eigenvalue = np.dot(x.T, np.dot(A, x)) / np.dot(x.T, x)
    return eigenvalue, x, k + 1


# Berechnung des Eigenwerts und Eigenvektors
eigenvalue, eigenvector, iterations = von_mises_iteration(A, x, tolerance, max_iterations)

# Überprüfung mit np.linalg.eig
eigenvalues, eigenvectors = np.linalg.eig(A)

# Ausgabe
print(f"Berechneter größter Eigenwert: {eigenvalue}")
print(f"Zugehöriger Eigenvektor: {eigenvector}")
print(f"Anzahl Iterationen: {iterations}")
print("\nÜberprüfung mit numpy.linalg.eig:")
print(f"Eigenwerte: {eigenvalues}")
print(f"Eigenvektoren: \n{eigenvectors}")

"""Aufgabe 6 b)"""

def scalar_matrix(n, scalar):
    """
    Create n×n scalar matrix (λI) - scalar multiple of identity matrix.
    Useful for characteristic equation calculations.
    
    Args:
        n: Size of the matrix (integer)
        scalar: Value to put on diagonal (float)
        
    Returns:
        2D list representing the scalar matrix
        
    Example:
        For n=2, scalar=3, returns [[3.0, 0.0], [0.0, 3.0]]
    """
    # Similar to identity_matrix but with scalar value instead of 1
    return [[scalar if i == j else 0.0 for j in range(n)] for i in range(n)]

def determinant(A):
    """
    Calculate matrix determinant using recursive expansion by first row.
    Uses Laplace expansion along first row for recursive calculation.
    
    Args:
        A: Input matrix (2D list of floats)
        
    Returns:
        float: Determinant value
        
    Note:
        Efficient for small matrices but not recommended for large ones
    """
    n = len(A)
    # Base cases for recursion
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]
        
    det = 0
    # Expand along first row
    for j in range(n):
        # Create submatrix by removing first row and current column
        submatrix = [[A[i][k] for k in range(n) if k != j] 
                    for i in range(1, n)]
        # Add term to determinant (with alternating sign)
        det += ((-1)**j) * A[0][j] * determinant(submatrix)
        
    return det



def characteristic_polynomial(A, lambda_val):
    """
    Evaluate characteristic polynomial det(A - λI) at given λ value.
    This polynomial's roots are the eigenvalues of A.
    
    Args:
        A: Input matrix (2D list of floats)
        lambda_val: Value at which to evaluate polynomial (float)
        
    Returns:
        float: Value of characteristic polynomial at lambda_val
    """
    n = len(A)
    # Create λI matrix
    lambda_matrix = scalar_matrix(n, lambda_val)
    # Compute A - λI
    diff_matrix = np.subtract(A, lambda_matrix)
    # Return determinant of A - λI
    return determinant(diff_matrix)

def trace(A):
    """
    Calculate matrix trace (sum of diagonal elements).
    
    Args:
        A: Input matrix (2D list of floats)
        
    Returns:
        float: Sum of diagonal elements
    """
    return sum(A[i][i] for i in range(len(A)))

