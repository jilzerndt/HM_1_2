#!/usr/bin/env python3
"""
COMPLETE HM2 EXAM PACKAGE - MASTER SCRIPT
All functions, methods, and utilities needed for the HM2 exam
Run this script to have everything available in one place
"""

# =============================================================================
# IMPORTS - Everything you need for the exam
# =============================================================================

import numpy as np
import sympy as sp
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import fsolve, minimize, minimize_scalar
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import CubicSpline, interp1d, lagrange
from scipy.linalg import qr, solve

# Custom utility functions - for exam obviously copy and paste!!
"""
BASICS: Support functions for evaluation, plotting, etc.
# =============================================================================
DATA ANALYSIS HELPERS
# load_and_preview_data: Load data from string format
# quick_error_analysis: Quick error analysis for exact vs approx values
# =============================================================================
MATRIX AND LINEAR ALGEBRA HELPERS
# evaluate_at_point: Evaluate symbolic expressions at given points
# quick_matrix_analysis: Quick matrix operations (det, inv, rank)
# solve_normal_equations: Solve normal equations for least squares (A^T A x = A^T b)
"""
from support_functions.basics import load_and_preview_data, quick_error_analysis, evaluate_at_point, quick_matrix_analysis, solve_normal_equations

# Setup matplotlib for better plots
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['grid.alpha'] = 0.3


