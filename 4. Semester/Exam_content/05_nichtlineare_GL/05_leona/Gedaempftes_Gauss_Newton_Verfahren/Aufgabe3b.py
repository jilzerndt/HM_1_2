import numpy as np
import matplotlib.pyplot as plt

# Gegebene Messdaten
t_data = np.array([0, 2, 4, 6, 8, 10])
H_data = np.array([52.9, 184, 426, 529, 499, 510])

# Modellfunktion: H(t) = A * (1 - exp(-K * t)) * exp(-G * t)
def model(t, params):
    A, G, K = params
    return A * (1 - np.exp(-K * t)) * np.exp(-G * t)

# Residuenfunktion
def residuals(params):
    return model(t_data, params) - H_data

# Jacobimatrix der Residuen
def jacobian(params):
    A, G, K = params
    J = np.zeros((len(t_data), 3))
    for i, t in enumerate(t_data):
        exp_Kt = np.exp(-K * t)
        exp_Gt = np.exp(-G * t)
        J[i, 0] = (1 - exp_Kt) * exp_Gt                      # ∂r/∂A
        J[i, 1] = -A * (1 - exp_Kt) * t * exp_Gt              # ∂r/∂G
        J[i, 2] = A * t * exp_Kt * exp_Gt                     # ∂r/∂K
    return J

# Gedämpftes Gauss-Newton-Verfahren mit Dämpfungsermittlung
def gauss_newton_damped(x0, tol=1e-7, max_iter=100, max_halvings=20):
    x = x0.copy()
    history = [x.copy()]
    min_halvings = 0

    for _ in range(max_iter):
        r = residuals(x)
        J = jacobian(x)
        delta = np.linalg.lstsq(J, -r, rcond=None)[0]

        damping = 0
        while damping <= max_halvings:
            x_new = x + delta / (2 ** damping)
            if np.linalg.norm(residuals(x_new)) < np.linalg.norm(r):
                break
            damping += 1

        if damping > min_halvings:
            min_halvings = damping

        x = x_new
        history.append(x.copy())

        if np.linalg.norm(delta) < tol:
            break

    return x, history, min_halvings

# Test: minimale Anzahl an erlaubten Schritthalbierungen, bei der das Verfahren überhaupt konvergiert
def try_gauss_newton_with_limit(x0, tol, max_halvings):
    x = x0.copy()
    for _ in range(100):
        r = residuals(x)
        J = jacobian(x)
        delta = np.linalg.lstsq(J, -r, rcond=None)[0]

        damping = 0
        while damping <= max_halvings:
            x_new = x + delta / (2 ** damping)
            if np.linalg.norm(residuals(x_new)) < np.linalg.norm(r):
                break
            damping += 1

        if damping > max_halvings:
            return False
        x = x_new
        if np.linalg.norm(delta) < tol:
            return True
    return False

# Starte mit gegebenen Startwerten
x0 = np.array([20, 450, 0.001])

# Führe Hauptlauf mit hoher Dämpfungsgrenze aus
params_fit, param_history, max_damping_used = gauss_newton_damped(x0)

# Suche minimale nötige max_halvings
minimal_required_halvings = None
for max_h in range(0, 50):
    if try_gauss_newton_with_limit(x0, tol=1e-7, max_halvings=max_h):
        minimal_required_halvings = max_h
        break

# Ausgabe
print("Gefundene Parameter A, G, K:", params_fit)
print("Maximal benötigte Dämpfung in einem Schritt:", max_damping_used)
print("Minimal nötige erlaubte Schritthalbierungen:", minimal_required_halvings)

# Plot
t_fine = np.linspace(0, 10, 300)
H_fit = model(t_fine, params_fit)

plt.figure(figsize=(8, 4))
plt.plot(t_data, H_data, 'ro', label='Messdaten')
plt.plot(t_fine, H_fit, 'b-', label='Gefittete Funktion')
plt.title("Messdaten und gefittete Funktion H(t)")
plt.xlabel("t")
plt.ylabel("H(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
