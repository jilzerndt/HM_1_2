import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math


def gauss_legendre_quadrature(f, a, b, n):
    """
    Gauss-Legendre quadrature
    
    Args:
        f: function to integrate
        a, b: integration limits
        n: number of points (1, 2, or 3)
    
    Returns:
        approximation of integral
    """
    # Transform to standard interval [-1, 1]
    def g(t):
        x = (b - a) * t / 2 + (a + b) / 2
        return f(x)
    
    if n == 1:
        # 1-point Gauss: x = 0, weight = 2
        result = 2 * g(0)
    elif n == 2:
        # 2-point Gauss
        x1, x2 = -1/np.sqrt(3), 1/np.sqrt(3)
        result = g(x1) + g(x2)
    elif n == 3:
        # 3-point Gauss
        x1, x2, x3 = -np.sqrt(0.6), 0, np.sqrt(0.6)
        w1, w2, w3 = 5/9, 8/9, 5/9
        result = w1 * g(x1) + w2 * g(x2) + w3 * g(x3)
    else:
        raise ValueError("Only n=1, 2, 3 supported for Gauss quadrature")
    
    # Transform back to [a, b]
    return (b - a) / 2 * result