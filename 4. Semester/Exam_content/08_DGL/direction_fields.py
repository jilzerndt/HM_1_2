import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches


def direction_field(f, x_range, y_range, nx=20, ny=20):
    """
    Plot direction field for ODE y' = f(x, y)
    
    Args:
        f: function f(x, y) defining the ODE
        x_range: tuple (x_min, x_max)
        y_range: tuple (y_min, y_max)
        nx, ny: number of grid points in x and y directions
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Calculate slopes
    DY = f(X, Y)
    DX = np.ones_like(DY)
    
    # Normalize arrows
    M = np.sqrt(DX**2 + DY**2)
    M[M == 0] = 1  # Avoid division by zero
    DX, DY = DX/M, DY/M
    
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, DX, DY, alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Direction Field for y' = f(x, y)")
    plt.grid(True, alpha=0.3)
    return plt.gca()


def quick_direction_field(ode_str, x_range, y_range, nx=15, ny=12):
    """
    Quick direction field plot
    
    Args:
        ode_str: ODE right-hand side as string
        x_range, y_range: tuples (min, max)
        nx, ny: grid resolution
    
    Example:
        quick_direction_field("x**2 + 0.1*y", (-2, 2), (-1, 3))
    """
    x_sym, y_sym = sp.symbols('x y')
    f_expr = sp.sympify(ode_str)
    f = sp.lambdify([x_sym, y_sym], f_expr, 'numpy')
    
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)
    
    DY = f(X, Y)
    DX = np.ones_like(DY)
    
    # Normalize arrows
    M = np.sqrt(DX**2 + DY**2)
    M[M == 0] = 1
    DX, DY = DX/M, DY/M
    
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, DX, DY, alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Direction Field for y' = {ode_str}")
    plt.grid(True, alpha=0.3)
    plt.show()

    
def direction_field_with_solutions(ode_str, x_range=(-2, 2), y_range=(-2, 2), 
                                 n_grid=15, solution_ics=None):
    """
    Direction field with solution curves
    """
    if solution_ics is None:
        solution_ics = [(x_range[0], 0), (x_range[0], 1), (x_range[0], -1)]
    
    x_sym, y_sym = sp.symbols('x y')
    f_expr = sp.sympify(ode_str)
    f_func = sp.lambdify([x_sym, y_sym], f_expr, 'numpy')
    
    # Create direction field
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)
    
    DY = f_func(X, Y)
    DX = np.ones_like(DY)
    
    # Normalize arrows
    M = np.sqrt(DX**2 + DY**2)
    M[M == 0] = 1
    DX_norm, DY_norm = DX/M, DY/M
    
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, DX_norm, DY_norm, alpha=0.6, color='gray')
    
    # Add solution curves
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    def ode_system(t, z):
        return f_func(t, z[0])
    
    for i, (x0, y0) in enumerate(solution_ics):
        try:
            sol = solve_ivp(ode_system, x_range, [y0], 
                          t_eval=np.linspace(x_range[0], x_range[1], 100),
                          rtol=1e-6)
            plt.plot(sol.t, sol.y[0], color=colors[i % len(colors)], 
                    linewidth=2, label=f'y({x0}) = {y0}')
            plt.plot(x0, y0, 'o', color=colors[i % len(colors)], markersize=8)
        except:
            print(f"Could not solve for initial condition ({x0}, {y0})")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Direction Field: y' = {ode_str}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.show()