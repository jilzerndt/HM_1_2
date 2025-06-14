import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches


def euler_method(f, x0, y0, h, n):
    """
    Euler's method for solving ODE y' = f(x, y)
    
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
        y[i + 1] = y[i] + h * f(x[i], y[i])
        x[i + 1] = x[i] + h
    
    return x, y

def midpoint_method(f, x0, y0, h, n):
    """
    Midpoint method for solving ODE y' = f(x, y)
    
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
        # Half step
        x_half = x[i] + h/2
        y_half = y[i] + (h/2) * f(x[i], y[i])
        
        # Full step using midpoint slope
        y[i + 1] = y[i] + h * f(x_half, y_half)
        x[i + 1] = x[i] + h
    
    return x, y

def modified_euler_method(f, x0, y0, h, n):
    """
    Modified Euler method (Heun's method) for solving ODE y' = f(x, y)
    
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
        # Predictor step (Euler)
        k1 = f(x[i], y[i])
        y_euler = y[i] + h * k1
        
        # Corrector step (average of slopes)
        k2 = f(x[i] + h, y_euler)
        y[i + 1] = y[i] + h * (k1 + k2) / 2
        x[i + 1] = x[i] + h
    
    return x, y