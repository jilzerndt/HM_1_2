import sympy as sp

# Define symbols
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Define functions
f1 = x1 + x2**2 - x3**2 - 13
f2 = sp.ln(x2 / 4) + sp.exp(0.5 * x3) - 1
f3 = (x2 - 3)**2 - x3**3 + 7

# partial derivatives
print("∂f1/∂x1 = ", sp.diff(f1, x1))
print("∂f1/∂x2 = ", sp.diff(f1, x2))
print("∂f1/∂x3 = ", sp.diff(f1, x3))

print("∂f2/∂x1 = ", sp.diff(f2, x1))
print("∂f2/∂x2 = ", sp.diff(f2, x2))
print("∂f2/∂x3 = ", sp.diff(f2, x3))

print("∂f3/∂x1 = ", sp.diff(f3, x1))
print("∂f3/∂x2 = ", sp.diff(f3, x2))
print("∂f3/∂x3 = ", sp.diff(f3, x3))

# Define matrix
f = sp.Matrix([f1, f2, f3])
print("Matrix f(x):")
sp.pprint(f)

# Compute Jacobian matrix
J = f.jacobian([x1, x2, x3])
print("Jacobian matrix J(x):")
sp.pprint(J)

# Linearization around x(0) = (1.5, 3, 2.5)^T
x0 = {x1: 1.5, x2: 3, x3: 2.5}

# Insert x0 into f and J
f_at_x0 = f.subs(x0)
J_at_x0 = J.subs(x0)

# Define vectors
x_vec = sp.Matrix([x1, x2, x3])
x0_vec = sp.Matrix([1.5, 3, 2.5])

# Linearized function
linear_approx = f_at_x0 + J_at_x0 * (x_vec - x0_vec)
print("Linearized function:")
sp.pprint(linear_approx)
