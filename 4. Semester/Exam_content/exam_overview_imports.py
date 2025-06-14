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

"""
PLOTTING: Functions for quick and effective plotting
# =============================================================================
# setup_plotting: Setup matplotlib for better plots
# quick_plot: Quick plotting function
# plot_data_and_fit: Plot data points and fitted curve
# plot_system_2d: Plot two 2D functions to visualize their intersection points
# plot_2d_system_complex: Plot complex 2D systems for intersection visualization
"""
from support_functions.plotting import setup_plotting, quick_plot, plot_data_and_fit, plot_system_2d, plot_2d_system_complex

"""
CHAPTER 5: NONLINEAR EQUATIONS (05_nichtlineare_GL)
# =============================================================================
PARTIELLE ABLEITUNGEN, JACOBI UND LINEARISIERUNG
# jacobian_symbolic: Symbolic Jacobian calculation
# jacobian_calculator: Jacobian matrix calculator
# quick_jacobian: Quick Jacobian evaluation
# linearize_function: Function linearization
# =============================================================================
# NEWTON METHOD FOR NONLINEAR SYSTEMS
# newton_method_systems: Newton method for systems
# damped_newton_method: Newton method with damping
# newton_2d_manual: Manual 2D Newton method
# quick_newton_2d: Quick 2D Newton solver
# =============================================================================
KR FUNCTIONS
# kr_newton_method_manual: Manual Newton method implementation
"""
from K05_nichtlineare_GL.partielle_ableitungen import jacobian_symbolic, jacobian_calculator, quick_jacobian, linearize_function
from K05_nichtlineare_GL.newton_nps import newton_method_systems, damped_newton_method, newton_2d_manual, quick_newton_2d

"""
CHAPTER 6: APPROXIMATION THEORY (06_ausgleichsrechnung)
# =============================================================================
AUSGLEICHSRECHNUNG
# linear_least_squares: Linear least squares solver
# quick_linear_fit: Quick linear regression
# linear_regression_manual: Manual linear regression
# gauss_newton_method: Gauss-Newton optimization
# polynomial_least_squares: Polynomial fitting
# quick_polyfit: Quick polynomial fitting
# =============================================================================
INTERPOLATION
# lagrange_interpolation: Lagrange interpolation
# quick_lagrange: Quick Lagrange interpolation
# lagrange_manual: Manual Lagrange interpolation
# natural_cubic_spline_coefficients: Cubic spline coefficients
# evaluate_cubic_spline: Evaluate cubic spline
# quick_spline: Quick spline interpolation
# cubic_spline_manual: Manual cubic spline
# =============================================================================
KR FUNCTIONS 
# kr_lagrange_interpolation: Lagrange interpolation implementation
# kr_natural_cubic_spline: Natural cubic spline implementation
# kr_linear_least_squares: Linear least squares fitting
# kr_gauss_newton: Gauss-Newton method for nonlinear fitting
"""
from K06_ausgleichsrechnung.ausgleichsrechnung import linear_least_squares, quick_linear_fit, linear_regression_manual, gauss_newton_method, polynomial_least_squares, quick_polyfit
from K06_ausgleichsrechnung.interpolation import lagrange_interpolation, quick_lagrange, lagrange_manual, natural_cubic_spline_coefficients, evaluate_cubic_spline, quick_spline, cubic_spline_manual

"""
CHAPTER 7: NUMERICAL INTEGRATION (07_numerische_integration)
# =============================================================================
# NUMERICAL INTEGRATION METHODS
# rectangle_rule: Rectangle rule integration
# trapezoidal_rule: Trapezoidal rule integration
# trapezoidal_rule_non_equidistant: Trapezoidal rule for non-equidistant points
# simpson_rule: Simpson's rule integration
# adaptive_simpson: Adaptive Simpson integration
# quick_integrate: Quick integration wrapper
# =============================================================================
# GAUSS QUADRATURE
# gauss_legendre_quadrature: Gauss-Legendre quadrature
# =============================================================================
# ROMBERG EXTRAPOLATION
# romberg_extrapolation: Romberg extrapolation
# quick_romberg_table: Quick Romberg table generation
# romberg_manual: Manual Romberg method
# =============================================================================
ERROR ANALYSIS AND CONVERGENCE
# error_analysis: Integration error analysis
# convergence_study: Convergence study for integration methods
# integration_comparison: Compare different integration methods
# =============================================================================
KR FUNCTIONS
# kr_numerical_integration: Numerical integration methods
# kr_romberg_extrapolation: Romberg extrapolation method
"""
from K07_numerische_integration.int_rules import rectangle_rule, trapezoidal_rule, trapezoidal_rule_non_equidistant, simpson_rule, adaptive_simpson, quick_integrate
from K07_numerische_integration.gauss_quad import gauss_legendre_quadrature
from K07_numerische_integration.romberg import romberg_extrapolation, quick_romberg_table, romberg_manual
from K07_numerische_integration.errors_convergence import error_analysis, convergence_study, integration_comparison

"""
CHAPTER 8: DIFFERENTIAL EQUATIONS (08_DGL)
# =============================================================================
DGL BASICS
# convert_higher_order_to_system: Convert higher-order ODEs to first-order systems
# stability_analysis: Stability analysis for ODE methods
# quick_ode_solve: Quick ODE solver wrapper
# ode_methods_comparison: Compare different ODE methods
# =============================================================================
DIRECTION FIELDS
# direction_field: Direction field plotting
# quick_direction_field: Quick direction field visualization
# direction_field_with_solutions: Direction field with solution curves
# =============================================================================
EULER METHODS
# euler_method: Euler method for ODEs
# midpoint_method: Midpoint method for ODEs
# modified_euler_method: Modified Euler method
# =============================================================================
RUNGE-KUTTA METHOD
# runge_kutta_4: 4th-order Runge-Kutta method
# =============================================================================
DGL/ODE SYSTEMS
# solve_ode_system_euler: Solve ODE systems with Euler method
# solve_ode_system_rk4: Solve ODE systems with RK4 method
# =============================================================================
KR FUNCTIONS
# kr_ode_methods: ODE solution methods
# kr_higher_order_to_system: Convert higher-order ODEs to systems
# kr_error_analysis: Error analysis for ODE methods
"""
from K08_DGL.dgl_basics import convert_higher_order_to_system, stability_analysis, quick_ode_solve, ode_methods_comparison
from K08_DGL.direction_fields import direction_field, quick_direction_field, direction_field_with_solutions
from K08_DGL.euler import euler_method, midpoint_method, modified_euler_method
from K08_DGL.runge_kutta import runge_kutta_4
from K08_DGL.dgl_systems import solve_ode_system_euler, solve_ode_system_rk4

"""
COMMON EXAM PROBLEMS - Ready-to-use examples and solutions
# =============================================================================
# atmospheric_pressure_example: Atmospheric pressure interpolation example
# temperature_density_example: Water density vs temperature example
# boeing_landing_example: Boeing landing problem (ODE example)

CHAPTER 5 EXAMPLES:
# ch5_circle_line_intersection: Circle-line intersection problem
# ch5_economic_equilibrium: Economic supply-demand equilibrium
# ch5_newton_method_example: Manual Newton method example

CHAPTER 6 EXAMPLES:
# ch6_atmospheric_pressure_interpolation: Atmospheric pressure interpolation
# ch6_water_density_fitting: Water density quadratic fitting
# ch6_cubic_spline_example: Natural cubic spline example

CHAPTER 7 EXAMPLES:
# ch7_basic_integration_rules: Compare integration rules
# ch7_romberg_example: Romberg extrapolation example
# ch7_gaussian_quadrature: Gaussian quadrature example

CHAPTER 8 EXAMPLES:
# ch8_boeing_landing_problem: Boeing landing ODE problem
# ch8_population_growth: Logistic population growth
# ch8_harmonic_oscillator: Harmonic oscillator system
# ch8_euler_vs_rk4_comparison: Compare Euler vs RK4 methods

CONVENIENCE FUNCTIONS:
# run_all_chapter5_examples: Run all Chapter 5 examples
# run_all_chapter6_examples: Run all Chapter 6 examples
# run_all_chapter7_examples: Run all Chapter 7 examples
# run_all_chapter8_examples: Run all Chapter 8 examples
# run_all_examples: Run ALL examples from all chapters
"""





