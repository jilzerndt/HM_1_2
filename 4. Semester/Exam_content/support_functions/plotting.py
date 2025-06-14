"""
Plotting functions for visualizing nonlinear systems
# plot_system_2d
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import interpolate
import pandas as pd

# =============================================================================
# QUICK SETUP FUNCTIONS
# =============================================================================

def setup_plotting():
    """Setup matplotlib for better plots"""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['grid.alpha'] = 0.3
    
def quick_plot(x, y, title="", xlabel="x", ylabel="y", label="", **kwargs):
    """Quick plotting function"""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=label, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_data_and_fit(x_data, y_data, x_fit, y_fit, title="Data and Fit"):
    """Plot data points and fitted curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ro', markersize=8, label='Data')
    plt.plot(x_fit, y_fit, 'b-', linewidth=2, label='Fit')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_system_2d(f1_func, f2_func, xlim=(-5, 5), ylim=(-5, 5), resolution=100):
    """
    Plot two 2D functions to visualize their intersection points
    """
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z1 = f1_func(X, Y)
    Z2 = f2_func(X, Y)
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z1, levels=[0], colors='blue', label='f1 = 0')
    plt.contour(X, Y, Z2, levels=[0], colors='red', label='f2 = 0')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Nonlinear System: Intersection of f1=0 and f2=0')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_2d_system_complex(f1_str, f2_str, xlim=(-5, 5), ylim=(-5, 5), resolution=400):
    """
    Plot two 2D functions to visualize intersection points (solutions)
    """
    x_sym, y_sym = sp.symbols('x y')
    f1_expr = sp.sympify(f1_str)
    f2_expr = sp.sympify(f2_str)
    f1_func = sp.lambdify([x_sym, y_sym], f1_expr, 'numpy')
    f2_func = sp.lambdify([x_sym, y_sym], f2_expr, 'numpy')
    
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    try:
        Z1 = f1_func(X, Y)
        Z2 = f2_func(X, Y)
        
        plt.figure(figsize=(12, 8))
        plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label=f'f₁ = {f1_str} = 0')
        plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label=f'f₂ = {f2_str} = 0')
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.title('Nonlinear System: Find intersection points')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.show()
        
        print("🎯 Visually estimate intersection points for starting values!")
        
    except Exception as e:
        print(f"Plotting error: {e}")