#!/usr/bin/env python3
"""
Chapter 8: Ordinary Differential Equations
Euler, Modified Euler, Midpoint, Runge-Kutta Methods, Systems of ODEs, Direction Fields
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches

# custom imports - copy parts from other files for exam
from dgl_basics import convert_higher_order_to_system, stability_analysis
from dgl_systems import solve_ode_system_euler, solve_ode_system_rk4
from runge_kutta import runge_kutta_4
from euler import euler_method, midpoint_method, modified_euler_method
from direction_fields import direction_field


# Example ODEs and systems
def example_ode_1(x, y):
    """Simple ODE: y' = x^2 + 0.1*y"""
    return x**2 + 0.1 * y

def example_ode_2(x, y):
    """Exponential decay: y' = -2.5*y"""
    return -2.5 * y

def example_system_1(x, z):
    """System: z'[0] = z[1], z'[1] = -z[0] (harmonic oscillator)"""
    return np.array([z[1], -z[0]])

def example_system_2(x, z):
    """Predator-prey system"""
    alpha, beta, gamma, delta = 1.0, 0.5, 0.75, 0.25
    prey, predator = z[0], z[1]
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return np.array([dprey_dt, dpredator_dt])

def exact_solution_1(x):
    """Exact solution for y' = -2.5*y, y(0) = 1"""
    return np.exp(-2.5 * x)

# Demo functions
def demo_direction_field():
    """Demo direction field plotting"""
    print("Direction Field Demo")
    print("-" * 30)
    
    def ode_func(x, y):
        return x**2 + 0.1 * y
    
    ax = direction_field(ode_func, (-2, 2), (-1, 3), nx=15, ny=12)
    
    # Add some solution curves
    colors = ['red', 'blue', 'green']
    y0_values = [0, 1, 2]
    
    for i, y0 in enumerate(y0_values):
        x_sol, y_sol = runge_kutta_4(ode_func, -2, y0, 0.1, 40)
        ax.plot(x_sol, y_sol, color=colors[i], linewidth=2, 
                label=f'Solution with y(-2) = {y0}')
    
    ax.legend()
    plt.title("Direction Field with Solution Curves")
    plt.show()

def demo_single_ode_methods():
    """Demo different methods for single ODE"""
    print("\nSingle ODE Methods Comparison")
    print("-" * 40)
    
    # Test ODE: y' = x^2 + 0.1*y with y(0) = 2
    def ode_func(x, y):
        return x**2 + 0.1 * y
    
    x0, y0 = 0, 2
    h = 0.2
    n = 25
    x_end = x0 + n * h
    
    print(f"ODE: y' = x² + 0.1y, y({x0}) = {y0}")
    print(f"Step size: h = {h}")
    print(f"Interval: [{x0}, {x_end}]")
    print()
    
    # Solve with different methods
    x_euler, y_euler = euler_method(ode_func, x0, y0, h, n)
    x_midpoint, y_midpoint = midpoint_method(ode_func, x0, y0, h, n)
    x_mod_euler, y_mod_euler = modified_euler_method(ode_func, x0, y0, h, n)
    x_rk4, y_rk4 = runge_kutta_4(ode_func, x0, y0, h, n)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(x_euler, y_euler, 'r.-', label='Euler', markersize=4)
    plt.plot(x_midpoint, y_midpoint, 'g.-', label='Midpoint', markersize=4)
    plt.plot(x_mod_euler, y_mod_euler, 'b.-', label='Modified Euler', markersize=4)
    plt.plot(x_rk4, y_rk4, 'k.-', label='Runge-Kutta 4', markersize=4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of ODE Solution Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error analysis (if exact solution known)
    # For demonstration, use RK4 with smaller step as "exact"
    x_ref, y_ref = runge_kutta_4(ode_func, x0, y0, h/10, n*10)
    y_ref_interp = np.interp(x_euler, x_ref, y_ref)
    
    errors_euler = np.abs(y_euler - y_ref_interp)
    errors_midpoint = np.abs(y_midpoint - y_ref_interp)
    errors_mod_euler = np.abs(y_mod_euler - y_ref_interp)
    errors_rk4 = np.abs(y_rk4 - y_ref_interp)
    
    plt.subplot(2, 1, 2)
    plt.semilogy(x_euler, errors_euler, 'r.-', label='Euler', markersize=4)
    plt.semilogy(x_midpoint, errors_midpoint, 'g.-', label='Midpoint', markersize=4)
    plt.semilogy(x_mod_euler, errors_mod_euler, 'b.-', label='Modified Euler', markersize=4)
    plt.semilogy(x_rk4, errors_rk4, 'k.-', label='Runge-Kutta 4', markersize=4)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Error Comparison (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final values
    print(f"Final values at x = {x_end}:")
    print(f"  Euler:         {y_euler[-1]:.6f}")
    print(f"  Midpoint:      {y_midpoint[-1]:.6f}")
    print(f"  Modified Euler: {y_mod_euler[-1]:.6f}")
    print(f"  Runge-Kutta 4: {y_rk4[-1]:.6f}")

def demo_system_solution():
    """Demo solving system of ODEs"""
    print("\nSystem of ODEs Demo")
    print("-" * 30)
    
    # Harmonic oscillator: y'' + y = 0
    # Converted to system: z1' = z2, z2' = -z1
    # Initial conditions: y(0) = 1, y'(0) = 0
    
    x0 = 0
    y0_vector = np.array([1.0, 0.0])  # [y, y']
    h = 0.1
    n = int(4 * np.pi / h)  # Integrate over 4π
    
    print("System: Harmonic Oscillator")
    print("z₁' = z₂, z₂' = -z₁")
    print(f"Initial conditions: z₁(0) = {y0_vector[0]}, z₂(0) = {y0_vector[1]}")
    print(f"Step size: h = {h}")
    
    # Solve with Euler and RK4
    x_euler, y_euler = solve_ode_system_euler(example_system_1, x0, y0_vector, h, n)
    x_rk4, y_rk4 = solve_ode_system_rk4(example_system_1, x0, y0_vector, h, n)
    
    # Exact solution
    x_exact = x_euler
    y_exact = np.cos(x_exact)  # y = cos(x)
    yp_exact = -np.sin(x_exact)  # y' = -sin(x)
    
    plt.figure(figsize=(15, 5))
    
    # Position comparison
    plt.subplot(1, 3, 1)
    plt.plot(x_exact, y_exact, 'k-', label='Exact', linewidth=2)
    plt.plot(x_euler, y_euler[:, 0], 'r--', label='Euler', alpha=0.7)
    plt.plot(x_rk4, y_rk4[:, 0], 'b--', label='RK4', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y (position)')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase portrait
    plt.subplot(1, 3, 2)
    plt.plot(y_exact, yp_exact, 'k-', label='Exact', linewidth=2)
    plt.plot(y_euler[:, 0], y_euler[:, 1], 'r--', label='Euler', alpha=0.7)
    plt.plot(y_rk4[:, 0], y_rk4[:, 1], 'b--', label='RK4', alpha=0.7)
    plt.xlabel('y (position)')
    plt.ylabel("y' (velocity)")
    plt.title('Phase Portrait')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Energy conservation (should be constant for harmonic oscillator)
    energy_exact = 0.5 * (yp_exact**2 + y_exact**2)
    energy_euler = 0.5 * (y_euler[:, 1]**2 + y_euler[:, 0]**2)
    energy_rk4 = 0.5 * (y_rk4[:, 1]**2 + y_rk4[:, 0]**2)
    
    plt.subplot(1, 3, 3)
    plt.plot(x_exact, energy_exact, 'k-', label='Exact', linewidth=2)
    plt.plot(x_euler, energy_euler, 'r--', label='Euler', alpha=0.7)
    plt.plot(x_rk4, energy_rk4, 'b--', label='RK4', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('Energy')
    plt.title('Energy Conservation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_stability():
    """Demo stability analysis"""
    print("\nStability Analysis Demo")
    print("-" * 30)
    
    # Test equation: y' = λy with λ = -2.5
    lambda_val = -2.5
    h_values = np.linspace(0.1, 1.5, 50)
    
    print(f"Test equation: y' = {lambda_val}y")
    print("Stability condition: |1 + hλ| < 1 for Euler method")
    print(f"Theoretical stability limit: h < {-2/lambda_val:.3f}")
    
    # Analyze stability for Euler method
    stability_euler = stability_analysis(euler_method, lambda_val, h_values)
    stability_rk4 = stability_analysis(runge_kutta_4, lambda_val, h_values)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(h_values, np.abs(stability_euler), 'r-', label='Euler', linewidth=2)
    plt.plot(h_values, np.abs(stability_rk4), 'b-', label='RK4', linewidth=2)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Stability threshold')
    plt.axvline(x=-2/lambda_val, color='r', linestyle=':', alpha=0.7, label='Euler limit')
    plt.xlabel('Step size h')
    plt.ylabel('|Growth factor|')
    plt.title('Stability Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 3)
    
    # Example of stable vs unstable behavior
    plt.subplot(1, 2, 2)
    
    def test_ode(x, y):
        return lambda_val * y
    
    x0, y0 = 0, 1
    n = 20
    
    # Stable case
    h_stable = 0.5
    x_stable, y_stable = euler_method(test_ode, x0, y0, h_stable, n)
    
    # Unstable case  
    h_unstable = 1.0
    x_unstable, y_unstable = euler_method(test_ode, x0, y0, h_unstable, n)
    
    # Exact solution
    x_exact = np.linspace(0, n*h_stable, 100)
    y_exact = np.exp(lambda_val * x_exact)
    
    plt.plot(x_exact, y_exact, 'k-', label='Exact', linewidth=2)
    plt.plot(x_stable, y_stable, 'g.-', label=f'Euler h={h_stable} (stable)')
    plt.plot(x_unstable, y_unstable, 'r.-', label=f'Euler h={h_unstable} (unstable)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Stable vs Unstable Behavior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_higher_order_conversion():
    """Demo converting higher-order ODE to system"""
    print("\nHigher-Order ODE Conversion Demo")
    print("-" * 40)
    
    # Example: y'' + 0.5*y' + 2*y = sin(x)
    # Convert to system: z1 = y, z2 = y'
    # z1' = z2, z2' = sin(x) - 2*z1 - 0.5*z2
    
    def coeffs_func(x):
        return np.array([2.0, 0.5])  # [a0, a1] for y'' + a1*y' + a0*y = f(x)
    
    def forcing_func(x):
        return np.sin(x)
    
    # Create system
    system_func = convert_higher_order_to_system(2, coeffs_func, forcing_func)
    
    # Initial conditions: y(0) = 1, y'(0) = 0
    x0 = 0
    y0_vector = np.array([1.0, 0.0])
    h = 0.05
    n = int(4*np.pi / h)
    
    print("Original ODE: y'' + 0.5y' + 2y = sin(x)")
    print("Initial conditions: y(0) = 1, y'(0) = 0")
    print("Converted to system of first-order ODEs")
    
    # Solve system
    x_vals, y_vals = solve_ode_system_rk4(system_func, x0, y0_vector, h, n)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals[:, 0], 'b-', label='y(x)', linewidth=2)
    plt.plot(x_vals, y_vals[:, 1], 'r-', label="y'(x)", linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution of Second-Order ODE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(y_vals[:, 0], y_vals[:, 1], 'g-', linewidth=2)
    plt.xlabel('y')
    plt.ylabel("y'")
    plt.title('Phase Portrait')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_with_scipy():
    """Compare our methods with SciPy's solve_ivp"""
    print("\nComparison with SciPy solve_ivp")
    print("-" * 40)
    
    def ode_func(x, y):
        return x**2 + 0.1 * y
    
    # For scipy, we need the function in the form f(t, y)
    def scipy_func(t, y):
        return ode_func(t, y[0])
    
    x0, y0 = 0, 2
    x_end = 3
    
    # Our RK4 method
    h = 0.1
    n = int(x_end / h)
    x_our, y_our = runge_kutta_4(ode_func, x0, y0, h, n)
    
    # SciPy's method
    sol = solve_ivp(scipy_func, [x0, x_end], [y0], dense_output=True)
    x_scipy = np.linspace(x0, x_end, len(x_our))
    y_scipy = sol.sol(x_scipy)[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_our, y_our, 'b.-', label='Our RK4', markersize=4)
    plt.plot(x_scipy, y_scipy, 'r-', label='SciPy solve_ivp', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison: Our RK4 vs SciPy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show difference
    error = np.abs(y_our - y_scipy)
    print(f"Maximum difference: {np.max(error):.2e}")
    print(f"Mean difference: {np.mean(error):.2e}")
    
    plt.figure(figsize=(8, 4))
    plt.semilogy(x_our, error, 'g.-')
    plt.xlabel('x')
    plt.ylabel('Absolute Difference')
    plt.title('Difference between Our RK4 and SciPy')
    plt.grid(True, alpha=0.3)
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Chapter 8: Ordinary Differential Equations")
    print("="*50)
    
    # Run all demos
    demo_direction_field()
    demo_single_ode_methods()
    demo_system_solution()
    demo_stability()
    demo_higher_order_conversion()
    compare_with_scipy()
    
    print("\nAvailable methods:")
    print("- euler_method()")
    print("- midpoint_method()")
    print("- modified_euler_method()")
    print("- runge_kutta_4()")
    print("- solve_ode_system_euler()")
    print("- solve_ode_system_rk4()")
    print("- direction_field()")
    print("- stability_analysis()")
    print("- convert_higher_order_to_system()")