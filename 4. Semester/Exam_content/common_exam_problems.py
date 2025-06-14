#!/usr/bin/env python3
"""
COMMON EXAM PROBLEMS - Organized by Chapter
Complete examples from exercises and typical exam questions
Each function is self-contained and ready to run
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve, minimize
from scipy import interpolate
from scipy.integrate import quad, solve_ivp
import pandas as pd

# =============================================================================
# CHAPTER 5: NONLINEAR EQUATIONS
# =============================================================================

def ch5_circle_line_intersection():
    """
    Classic problem: Find intersection of circle and line
    Circle: x² + y² = 25
    Line: x + y = 7
    """
    print("=== Chapter 5: Circle-Line Intersection ===")
    print("Circle: x² + y² = 25")
    print("Line: x + y = 7")
    
    def system(vars):
        x, y = vars
        f1 = x**2 + y**2 - 25  # Circle equation
        f2 = x + y - 7         # Line equation
        return [f1, f2]
    
    # Starting values
    sol1 = fsolve(system, [1, 6])
    sol2 = fsolve(system, [6, 1])
    
    print(f"Solution 1: x = {sol1[0]:.3f}, y = {sol1[1]:.3f}")
    print(f"Solution 2: x = {sol2[0]:.3f}, y = {sol2[1]:.3f}")
    
    # Verification
    print("Verification:")
    for i, sol in enumerate([sol1, sol2], 1):
        x, y = sol
        circle_check = x**2 + y**2
        line_check = x + y
        print(f"Sol {i}: x²+y² = {circle_check:.3f}, x+y = {line_check:.3f}")
    
    return sol1, sol2

def ch5_economic_equilibrium():
    """
    Economic equilibrium problem
    Supply: p = 0.1q² + 2q + 5
    Demand: p = -0.05q² + 8q + 20
    Find equilibrium (where supply = demand)
    """
    print("=== Chapter 5: Economic Equilibrium ===")
    print("Supply: p = 0.1q² + 2q + 5")
    print("Demand: p = -0.05q² + 8q + 20")
    
    def equilibrium(q):
        supply = 0.1*q**2 + 2*q + 5
        demand = -0.05*q**2 + 8*q + 20
        return supply - demand
    
    q_eq = fsolve(equilibrium, 10)[0]
    p_eq = 0.1*q_eq**2 + 2*q_eq + 5
    
    print(f"Equilibrium quantity: q = {q_eq:.3f}")
    print(f"Equilibrium price: p = {p_eq:.3f}")
    
    return q_eq, p_eq

def ch5_newton_method_example():
    """
    Manual Newton method example: f(x) = x³ - 2x - 5
    """
    print("=== Chapter 5: Newton Method Example ===")
    print("f(x) = x³ - 2x - 5")
    print("f'(x) = 3x² - 2")
    
    def f(x):
        return x**3 - 2*x - 5
    
    def df(x):
        return 3*x**2 - 2
    
    # Newton iteration
    x = 2.0  # Starting value
    print(f"x₀ = {x}")
    
    for i in range(5):
        x_new = x - f(x)/df(x)
        print(f"x_{i+1} = {x_new:.6f}, f(x_{i+1}) = {f(x_new):.8f}")
        x = x_new
    
    return x

# =============================================================================
# CHAPTER 6: APPROXIMATION THEORY
# =============================================================================

def ch6_atmospheric_pressure_interpolation():
    """
    Atmospheric pressure vs altitude interpolation
    Find pressure at 3750m using Lagrange interpolation
    """
    print("=== Chapter 6: Atmospheric Pressure Interpolation ===")
    heights = np.array([0, 2500, 5000, 10000])
    pressures = np.array([1013, 747, 540, 226])
    
    print("Data points:")
    for h, p in zip(heights, pressures):
        print(f"h = {h:4d}m, p = {p:4d} hPa")
    
    # Lagrange interpolation at 3750m
    target_height = 3750
    
    # Manual Lagrange calculation
    result = 0
    n = len(heights)
    
    for i in range(n):
        # Calculate Li(x)
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (target_height - heights[j]) / (heights[i] - heights[j])
        result += pressures[i] * Li
    
    print(f"Pressure at {target_height}m: {result:.1f} hPa")
    
    # Using scipy for comparison
    f = interpolate.lagrange(heights, pressures)
    scipy_result = f(target_height)
    print(f"SciPy verification: {scipy_result:.1f} hPa")
    
    return result

def ch6_water_density_fitting():
    """
    Water density vs temperature - quadratic fit
    """
    print("=== Chapter 6: Water Density vs Temperature ===")
    temperatures = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    densities = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4])
    
    # Quadratic fit: ρ(T) = aT² + bT + c
    coeffs = np.polyfit(temperatures, densities, 2)
    a, b, c = coeffs
    
    print(f"Quadratic fit: ρ(T) = {a:.6f}T² + {b:.6f}T + {c:.6f}")
    
    # Calculate R²
    y_pred = np.polyval(coeffs, temperatures)
    ss_res = np.sum((densities - y_pred)**2)
    ss_tot = np.sum((densities - np.mean(densities))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"R² = {r_squared:.6f}")
    
    # Predict density at 25°C
    temp_25 = 25
    density_25 = np.polyval(coeffs, temp_25)
    print(f"Density at 25°C: {density_25:.2f} kg/m³")
    
    return coeffs, r_squared

def ch6_cubic_spline_example():
    """
    Natural cubic spline example
    """
    print("=== Chapter 6: Cubic Spline Example ===")
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 4, 9])  # y = x²
    
    print("Data points:", list(zip(x, y)))
    
    # Create cubic spline
    spline = interpolate.CubicSpline(x, y, bc_type='natural')
    
    # Evaluate at intermediate points
    x_eval = np.array([0.5, 1.5, 2.5])
    y_eval = spline(x_eval)
    
    print("Spline evaluations:")
    for xi, yi in zip(x_eval, y_eval):
        print(f"S({xi}) = {yi:.3f}")
    
    return spline

# =============================================================================
# CHAPTER 7: NUMERICAL INTEGRATION
# =============================================================================

def ch7_basic_integration_rules():
    """
    Compare different integration rules on f(x) = x²
    Integral from 0 to 2 (exact = 8/3 ≈ 2.667)
    """
    print("=== Chapter 7: Integration Rules Comparison ===")
    print("f(x) = x², integral from 0 to 2")
    print("Exact value = 8/3 ≈ 2.6667")
    
    def f(x):
        return x**2
    
    a, b = 0, 2
    exact = 8/3
    
    # Rectangle rule (n=4)
    n = 4
    h = (b - a) / n
    x_rect = np.linspace(a, b-h, n) + h/2  # Midpoints
    rect_approx = h * np.sum(f(x_rect))
    
    # Trapezoidal rule (n=4)
    x_trap = np.linspace(a, b, n+1)
    trap_approx = h/2 * (f(a) + 2*np.sum(f(x_trap[1:-1])) + f(b))
    
    # Simpson's rule (n=4, must be even)
    simp_approx = h/3 * (f(a) + 4*np.sum(f(x_trap[1::2])) + 2*np.sum(f(x_trap[2:-1:2])) + f(b))
    
    print(f"Rectangle rule: {rect_approx:.6f}, error: {abs(rect_approx - exact):.6f}")
    print(f"Trapezoidal rule: {trap_approx:.6f}, error: {abs(trap_approx - exact):.6f}")
    print(f"Simpson's rule: {simp_approx:.6f}, error: {abs(simp_approx - exact):.6f}")
    
    return rect_approx, trap_approx, simp_approx

def ch7_romberg_example():
    """
    Romberg extrapolation example
    """
    print("=== Chapter 7: Romberg Extrapolation ===")
    print("f(x) = 1/(1+x²), integral from 0 to 1")
    print("Exact value = π/4 ≈ 0.7854")
    
    def f(x):
        return 1/(1 + x**2)
    
    a, b = 0, 1
    exact = np.pi/4
    
    # Romberg table (4x4)
    R = np.zeros((4, 4))
    
    # First column: Trapezoidal rule with h, h/2, h/4, h/8
    for i in range(4):
        n = 2**i
        h = (b - a) / n
        if i == 0:
            R[i, 0] = h/2 * (f(a) + f(b))
        else:
            # Use recursive formula
            sum_new = sum(f(a + (2*k-1)*h) for k in range(1, n//2 + 1))
            R[i, 0] = R[i-1, 0]/2 + h * sum_new
    
    # Fill the rest of the table
    for j in range(1, 4):
        for i in range(j, 4):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    
    print("Romberg Table:")
    print("Row\\Col   0        1        2        3")
    for i in range(4):
        row_str = f"  {i}   "
        for j in range(i+1):
            row_str += f"{R[i, j]:.6f} "
        print(row_str)
    
    print(f"Best approximation: {R[3, 3]:.6f}")
    print(f"Error: {abs(R[3, 3] - exact):.8f}")
    
    return R

def ch7_gaussian_quadrature():
    """
    Gaussian quadrature example
    """
    print("=== Chapter 7: Gaussian Quadrature ===")
    print("f(x) = e^x, integral from -1 to 1")
    print("Exact value = e - 1/e ≈ 2.3504")
    
    def f(x):
        return np.exp(x)
    
    exact = np.exp(1) - np.exp(-1)
    
    # 2-point Gauss-Legendre
    # Points: ±1/√3, Weights: 1, 1
    x1, x2 = -1/np.sqrt(3), 1/np.sqrt(3)
    w1, w2 = 1, 1
    
    gauss_2pt = w1*f(x1) + w2*f(x2)
    
    print(f"2-point Gauss: {gauss_2pt:.6f}, error: {abs(gauss_2pt - exact):.6f}")
    
    # Compare with trapezoidal (2 points)
    trap_2pt = 1 * (f(-1) + f(1))
    print(f"Trapezoidal: {trap_2pt:.6f}, error: {abs(trap_2pt - exact):.6f}")
    
    return gauss_2pt

# =============================================================================
# CHAPTER 8: DIFFERENTIAL EQUATIONS
# =============================================================================

def ch8_boeing_landing_problem():
    """
    Boeing 737-200 landing problem
    m = 97,000 kg, v₀ = 100 m/s
    F = -5v² - 570,000
    """
    print("=== Chapter 8: Boeing Landing Problem ===")
    print("m = 97,000 kg, v₀ = 100 m/s")
    print("F = -5v² - 570,000")
    print("ODE: m(dv/dt) = -5v² - 570,000")
    
    m = 97000
    v0 = 100
    
    def boeing_ode(t, y):
        v = y[0]
        F = -5*v**2 - 570000
        dvdt = F/m
        return [dvdt]
    
    # Solve for 20 seconds
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 100)
    
    sol = solve_ivp(boeing_ode, t_span, [v0], t_eval=t_eval, method='RK45')
    
    # Find stopping time (when v ≈ 0)
    stop_idx = np.where(sol.y[0] <= 1)[0]
    if len(stop_idx) > 0:
        stop_time = sol.t[stop_idx[0]]
        print(f"Stopping time: {stop_time:.2f} seconds")
    
    # Calculate distance (integrate velocity)
    distances = np.cumsum(sol.y[0][:-1] * np.diff(sol.t))
    total_distance = distances[-1] if len(distances) > 0 else 0
    print(f"Stopping distance: {total_distance:.0f} meters")
    
    return sol

def ch8_population_growth():
    """
    Logistic population growth model
    dp/dt = rp(1 - p/K)
    r = 0.1, K = 1000, p₀ = 50
    """
    print("=== Chapter 8: Logistic Population Growth ===")
    print("dp/dt = rp(1 - p/K)")
    print("r = 0.1, K = 1000, p₀ = 50")
    
    r = 0.1
    K = 1000
    p0 = 50
    
    def logistic_ode(t, y):
        p = y[0]
        dpdt = r * p * (1 - p/K)
        return [dpdt]
    
    # Solve for 50 time units
    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 100)
    
    sol = solve_ivp(logistic_ode, t_span, [p0], t_eval=t_eval)
    
    # Analytical solution for comparison
    def analytical_solution(t):
        return K / (1 + ((K - p0)/p0) * np.exp(-r*t))
    
    p_analytical = analytical_solution(t_eval)
    
    print(f"Final population (numerical): {sol.y[0][-1]:.1f}")
    print(f"Final population (analytical): {p_analytical[-1]:.1f}")
    print(f"Carrying capacity: {K}")
    
    return sol, p_analytical

def ch8_harmonic_oscillator():
    """
    Harmonic oscillator: y'' + ω²y = 0
    Convert to system: y₁ = y, y₂ = y'
    """
    print("=== Chapter 8: Harmonic Oscillator ===")
    print("y'' + ω²y = 0, ω = 2")
    print("Initial conditions: y(0) = 1, y'(0) = 0")
    
    omega = 2
    
    def harmonic_system(t, z):
        y, dydt = z
        d2ydt2 = -omega**2 * y
        return [dydt, d2ydt2]
    
    # Initial conditions
    y0, dydt0 = 1, 0
    
    # Solve for one period
    T = 2*np.pi/omega
    t_span = (0, T)
    t_eval = np.linspace(0, T, 100)
    
    sol = solve_ivp(harmonic_system, t_span, [y0, dydt0], t_eval=t_eval)
    
    # Analytical solution
    y_analytical = np.cos(omega * t_eval)
    
    print(f"Period: {T:.3f}")
    print(f"Final position (numerical): {sol.y[0][-1]:.6f}")
    print(f"Final position (analytical): {y_analytical[-1]:.6f}")
    print("Should return to initial position (1.0)")
    
    return sol, y_analytical

def ch8_euler_vs_rk4_comparison():
    """
    Compare Euler and RK4 methods on dy/dt = -2y + 1, y(0) = 0
    Analytical solution: y(t) = 0.5(1 - e^(-2t))
    """
    print("=== Chapter 8: Euler vs RK4 Comparison ===")
    print("dy/dt = -2y + 1, y(0) = 0")
    print("Analytical: y(t) = 0.5(1 - e^(-2t))")
    
    def ode_func(t, y):
        return -2*y + 1
    
    def analytical_solution(t):
        return 0.5*(1 - np.exp(-2*t))
    
    # Parameters
    t0, tf = 0, 2
    y0 = 0
    h = 0.1
    
    # Euler method
    t_euler = np.arange(t0, tf + h, h)
    y_euler = np.zeros_like(t_euler)
    y_euler[0] = y0
    
    for i in range(len(t_euler) - 1):
        y_euler[i+1] = y_euler[i] + h * ode_func(t_euler[i], y_euler[i])
    
    # RK4 method
    y_rk4 = np.zeros_like(t_euler)
    y_rk4[0] = y0
    
    for i in range(len(t_euler) - 1):
        t_i = t_euler[i]
        y_i = y_rk4[i]
        
        k1 = h * ode_func(t_i, y_i)
        k2 = h * ode_func(t_i + h/2, y_i + k1/2)
        k3 = h * ode_func(t_i + h/2, y_i + k2/2)
        k4 = h * ode_func(t_i + h, y_i + k3)
        
        y_rk4[i+1] = y_i + (k1 + 2*k2 + 2*k3 + k4)/6
    
    # Analytical solution
    y_analytical = analytical_solution(t_euler)
    
    # Final errors
    error_euler = abs(y_euler[-1] - y_analytical[-1])
    error_rk4 = abs(y_rk4[-1] - y_analytical[-1])
    
    print(f"At t = {tf}:")
    print(f"Analytical: {y_analytical[-1]:.6f}")
    print(f"Euler: {y_euler[-1]:.6f}, error: {error_euler:.6f}")
    print(f"RK4: {y_rk4[-1]:.6f}, error: {error_rk4:.6f}")
    
    return t_euler, y_euler, y_rk4, y_analytical

# =============================================================================
# CONVENIENCE FUNCTIONS - Run ALL Examples
# =============================================================================

def run_all_chapter5_examples():
    """Run all Chapter 5 examples"""
    print("=" * 60)
    print("CHAPTER 5: NONLINEAR EQUATIONS - ALL EXAMPLES")
    print("=" * 60)
    
    ch5_circle_line_intersection()
    print()
    ch5_economic_equilibrium()
    print()
    ch5_newton_method_example()
    print()

def run_all_chapter6_examples():
    """Run all Chapter 6 examples"""
    print("=" * 60)
    print("CHAPTER 6: APPROXIMATION THEORY - ALL EXAMPLES")
    print("=" * 60)
    
    ch6_atmospheric_pressure_interpolation()
    print()
    ch6_water_density_fitting()
    print()
    ch6_cubic_spline_example()
    print()

def run_all_chapter7_examples():
    """Run all Chapter 7 examples"""
    print("=" * 60)
    print("CHAPTER 7: NUMERICAL INTEGRATION - ALL EXAMPLES")
    print("=" * 60)
    
    ch7_basic_integration_rules()
    print()
    ch7_romberg_example()
    print()
    ch7_gaussian_quadrature()
    print()

def run_all_chapter8_examples():
    """Run all Chapter 8 examples"""
    print("=" * 60)
    print("CHAPTER 8: DIFFERENTIAL EQUATIONS - ALL EXAMPLES")
    print("=" * 60)
    
    ch8_boeing_landing_problem()
    print()
    ch8_population_growth()
    print()
    ch8_harmonic_oscillator()
    print()
    ch8_euler_vs_rk4_comparison()
    print()

def run_all_examples():
    """Run ALL examples from all chapters"""
    print("🎯 RUNNING ALL HM2 EXAM EXAMPLES")
    print("=" * 80)
    
    run_all_chapter5_examples()
    run_all_chapter6_examples()
    run_all_chapter7_examples()
    run_all_chapter8_examples()
    
    print("✅ ALL EXAMPLES COMPLETED!")

# =============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# =============================================================================

def atmospheric_pressure_example():
    """Legacy function - redirects to new chapter-organized version"""
    return ch6_atmospheric_pressure_interpolation()

def temperature_density_example():
    """Legacy function - redirects to new chapter-organized version"""
    return ch6_water_density_fitting()

def boeing_landing_example():
    """Legacy function - redirects to new chapter-organized version"""
    return ch8_boeing_landing_problem()

if __name__ == "__main__":
    # Run a few examples when script is executed directly
    print("Running sample examples...")
    ch5_circle_line_intersection()
    print()
    ch6_atmospheric_pressure_interpolation()
    print()
    ch7_basic_integration_rules()
    print()
    ch8_population_growth()