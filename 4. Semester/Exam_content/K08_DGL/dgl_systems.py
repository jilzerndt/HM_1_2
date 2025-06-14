import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches


def solve_ode_system_euler(f_system, x0, y0_vector, h, n):
    """
    Euler's method for systems of ODEs
    
    Args:
        f_system: function that returns dy/dx as a vector
        x0: initial x value
        y0_vector: initial y values as numpy array
        h: step size
        n: number of steps
    
    Returns:
        x_values: array of x values
        y_values: array of y vectors (each row is a step)
    """
    dim = len(y0_vector)
    x = np.zeros(n + 1)
    y = np.zeros((n + 1, dim))
    
    x[0] = x0
    y[0] = y0_vector
    
    for i in range(n):
        dy = f_system(x[i], y[i])
        y[i + 1] = y[i] + h * dy
        x[i + 1] = x[i] + h
    
    return x, y

def solve_ode_system_rk4(f_system, x0, y0_vector, h, n):
    """
    4th-order Runge-Kutta method for systems of ODEs
    
    Args:
        f_system: function that returns dy/dx as a vector
        x0: initial x value
        y0_vector: initial y values as numpy array
        h: step size
        n: number of steps
    
    Returns:
        x_values: array of x values
        y_values: array of y vectors (each row is a step)
    """
    dim = len(y0_vector)
    x = np.zeros(n + 1)
    y = np.zeros((n + 1, dim))
    
    x[0] = x0
    y[0] = y0_vector
    
    for i in range(n):
        k1 = f_system(x[i], y[i])
        k2 = f_system(x[i] + h/2, y[i] + h*k1/2)
        k3 = f_system(x[i] + h/2, y[i] + h*k2/2)
        k4 = f_system(x[i] + h, y[i] + h*k3)
        
        y[i + 1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x[i + 1] = x[i] + h
    
    return x, y