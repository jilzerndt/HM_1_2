import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches



def convert_higher_order_to_system(order, coeffs_func, forcing_func):
    """
    Convert higher-order ODE to first-order system
    
    For equation: y^(n) + a_{n-1}*y^(n-1) + ... + a_1*y' + a_0*y = f(x)
    
    Args:
        order: order of the ODE
        coeffs_func: function that returns coefficients [a_0, a_1, ..., a_{n-1}]
        forcing_func: forcing function f(x)
    
    Returns:
        system_func: function for the equivalent first-order system
    """
    def system_func(x, z):
        """
        z[0] = y
        z[1] = y'
        z[2] = y''
        ...
        z[n-1] = y^(n-1)
        """
        dz = np.zeros(order)
        
        # z'[i] = z[i+1] for i = 0, 1, ..., n-2
        for i in range(order - 1):
            dz[i] = z[i + 1]
        
        # z'[n-1] = y^(n) = f(x) - a_{n-1}*y^(n-1) - ... - a_1*y' - a_0*y
        coeffs = coeffs_func(x)
        dz[order - 1] = forcing_func(x) - np.sum(coeffs * z)
        
        return dz
    
    return system_func


def stability_analysis(method_func, test_equation_lambda, h_values):
    """
    Analyze stability of numerical method for test equation y' = λy
    
    Args:
        method_func: numerical method function
        test_equation_lambda: λ parameter for test equation
        h_values: array of step sizes to test
    
    Returns:
        stability_factors: growth factors for each h
    """
    def test_ode(x, y):
        return test_equation_lambda * y
    
    stability_factors = []
    
    for h in h_values:
        # Take one step from y=1
        _, y_vals = method_func(test_ode, 0, 1, h, 1)
        growth_factor = y_vals[1] / y_vals[0]
        stability_factors.append(growth_factor)
    
    return np.array(stability_factors)


def quick_ode_solve(ode_str, x0, y0, x_end, method='rk4', n=100, plot=True):
    """
    Quick ODE solver
    
    Args:
        ode_str: ODE as string "dy/dx = f(x,y)" - just provide the f(x,y) part
        x0, y0: initial conditions
        x_end: end point
        method: 'euler', 'midpoint', 'modified_euler', 'rk4'
        n: number of steps
        plot: whether to plot result
    
    Example:
        x, y = quick_ode_solve("x**2 + 0.1*y", 0, 2, 5, 'rk4', 50)
    """
    x_sym, y_sym = sp.symbols('x y')
    f_expr = sp.sympify(ode_str)
    f = sp.lambdify([x_sym, y_sym], f_expr, 'numpy')
    
    h = (x_end - x0) / n
    x_vals = np.zeros(n + 1)
    y_vals = np.zeros(n + 1)
    x_vals[0], y_vals[0] = x0, y0
    
    for i in range(n):
        if method == 'euler':
            y_vals[i+1] = y_vals[i] + h * f(x_vals[i], y_vals[i])
        
        elif method == 'midpoint':
            x_half = x_vals[i] + h/2
            y_half = y_vals[i] + (h/2) * f(x_vals[i], y_vals[i])
            y_vals[i+1] = y_vals[i] + h * f(x_half, y_half)
        
        elif method == 'modified_euler':
            k1 = f(x_vals[i], y_vals[i])
            y_euler = y_vals[i] + h * k1
            k2 = f(x_vals[i] + h, y_euler)
            y_vals[i+1] = y_vals[i] + h * (k1 + k2) / 2
        
        elif method == 'rk4':
            k1 = f(x_vals[i], y_vals[i])
            k2 = f(x_vals[i] + h/2, y_vals[i] + h*k1/2)
            k3 = f(x_vals[i] + h/2, y_vals[i] + h*k2/2)
            k4 = f(x_vals[i] + h, y_vals[i] + h*k3)
            y_vals[i+1] = y_vals[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        x_vals[i+1] = x_vals[i] + h
    
    if plot:
        quick_plot(x_vals, y_vals, f"ODE Solution using {method}", "x", "y")
    
    print(f"Final value: y({x_end}) ≈ {y_vals[-1]:.6f}")
    return x_vals, y_vals


def ode_methods_comparison(ode_str, x0, y0, x_end, n=10, show_steps=False):
    """
    Compare all ODE methods with error analysis
    """
    x_sym, y_sym = sp.symbols('x y')
    f_expr = sp.sympify(ode_str)
    f_func = sp.lambdify([x_sym, y_sym], f_expr, 'numpy')
    
    print(f"ODE Methods Comparison")
    print("="*30)
    print(f"ODE: y' = {ode_str}")
    print(f"Initial condition: y({x0}) = {y0}")
    print(f"Solve on interval [{x0}, {x_end}] with {n} steps")
    
    h = (x_end - x0) / n
    print(f"Step size: h = {h}")
    print()
    
    # Initialize arrays
    x_vals = np.linspace(x0, x_end, n + 1)
    
    methods = {}
    
    # Euler Method
    y_euler = np.zeros(n + 1)
    y_euler[0] = y0
    
    for i in range(n):
        y_euler[i+1] = y_euler[i] + h * f_func(x_vals[i], y_euler[i])
    
    methods['Euler'] = y_euler.copy()
    
    # Midpoint Method
    y_midpoint = np.zeros(n + 1)
    y_midpoint[0] = y0
    
    for i in range(n):
        x_half = x_vals[i] + h/2
        y_half = y_midpoint[i] + (h/2) * f_func(x_vals[i], y_midpoint[i])
        y_midpoint[i+1] = y_midpoint[i] + h * f_func(x_half, y_half)
    
    methods['Midpoint'] = y_midpoint.copy()
    
    # Modified Euler
    y_mod_euler = np.zeros(n + 1)
    y_mod_euler[0] = y0
    
    for i in range(n):
        k1 = f_func(x_vals[i], y_mod_euler[i])
        y_euler_pred = y_mod_euler[i] + h * k1
        k2 = f_func(x_vals[i] + h, y_euler_pred)
        y_mod_euler[i+1] = y_mod_euler[i] + h * (k1 + k2) / 2
    
    methods['Modified Euler'] = y_mod_euler.copy()
    
    # Runge-Kutta 4
    y_rk4 = np.zeros(n + 1)
    y_rk4[0] = y0
    
    for i in range(n):
        k1 = f_func(x_vals[i], y_rk4[i])
        k2 = f_func(x_vals[i] + h/2, y_rk4[i] + h*k1/2)
        k3 = f_func(x_vals[i] + h/2, y_rk4[i] + h*k2/2)
        k4 = f_func(x_vals[i] + h, y_rk4[i] + h*k3)
        y_rk4[i+1] = y_rk4[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    methods['Runge-Kutta 4'] = y_rk4.copy()
    
    # Show final values
    print("Final values at x =", x_end)
    for name, y_vals in methods.items():
        print(f"  {name:15s}: y({x_end}) = {y_vals[-1]:.8f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green', 'black']
    for i, (name, y_vals) in enumerate(methods.items()):
        plt.plot(x_vals, y_vals, 'o-', color=colors[i], label=name, markersize=4)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ODE Methods Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error comparison (using RK4 as reference)
    plt.subplot(1, 2, 2)
    rk4_vals = methods['Runge-Kutta 4']
    
    for i, (name, y_vals) in enumerate(methods.items()):
        if name != 'Runge-Kutta 4':
            errors = np.abs(y_vals - rk4_vals)
            plt.semilogy(x_vals, errors, 'o-', color=colors[i], label=f'{name} vs RK4', markersize=4)
    
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Error vs Runge-Kutta 4')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return methods, x_vals