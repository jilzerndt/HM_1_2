import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches


def runge_kutta_4(f, x0, y0, h, n):
    """
    Classical 4th-order Runge-Kutta method for solving ODE y' = f(x, y)
    
    Args:
        f: function f(x, y) defining the ODE
        x0: initial x value
        y0: initial y value
        h: step size
        n: number of steps
    
    Returns:
        x_values, y_values: arrays of solution points
    """
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    
    x[0], y[0] = x0, y0
    
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h*k1/2)
        k3 = f(x[i] + h/2, y[i] + h*k2/2)
        k4 = f(x[i] + h, y[i] + h*k3)
        
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x[i + 1] = x[i] + h
    
    return x, y