#!/usr/bin/env python3
"""
Exam Utility Functions
Quick reference functions for common operations during the exam
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve
from scipy import interpolate
import pandas as pd




# =============================================================================
# COMMON EXAM FUNCTIONS
# =============================================================================

def atmospheric_pressure_example():
    """Example from Serie 4: Atmospheric pressure interpolation"""
    heights = [0, 2500, 5000, 10000]  # NaN at 3750 removed
    pressures = [1013, 747, 540, 226]
    
    # Find missing value at 3750m using Lagrange interpolation
    missing_pressure = quick_lagrange(heights, pressures, 3750)
    print(f"Atmospheric pressure at 3750m: {missing_pressure:.1f} hPa")
    
    return missing_pressure

def temperature_density_example():
    """Example from Serie 6: Water density vs temperature"""
    temperatures = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    densities = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4])
    
    # Quadratic fit: f(T) = aT² + bT + c
    coeffs = quick_polyfit(temperatures, densities, 2, plot=True)
    
    print("Water density as function of temperature:")
    print(f"ρ(T) = {coeffs[0]:.6f}T² + {coeffs[1]:.6f}T + {coeffs[2]:.6f}")
    
    return coeffs

def boeing_landing_example():
    """Example from Serie 10 & 13: Boeing landing problem"""
    print("Boeing 737-200 Landing Problem")
    print("m = 97,000 kg, v₀ = 100 m/s")
    print("F = -5v² - 570,000")
    print("ODE: m(dv/dt) = -5v² - 570,000")
    
    # Convert to first-order system
    def boeing_system(t, z):
        # z[0] = x (position), z[1] = v (velocity)
        m = 97000
        F = -5 * z[1]**2 - 570000
        return np.array([z[1], F/m])
    
    # This would be solved numerically
    print("Convert to system: dx/dt = v, dv/dt = F/m")
    return boeing_system

