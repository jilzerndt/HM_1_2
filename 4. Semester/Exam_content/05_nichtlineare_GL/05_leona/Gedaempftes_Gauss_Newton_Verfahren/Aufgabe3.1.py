import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Gegebene Messdaten
t_data = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
U_data = np.array([39.55, 46.55, 50.13, 51.75, 55.25, 56.79, 56.78, 59.19, 57.76, 59.39, 60.08])

# Plot der Messdaten
plt.figure()
plt.plot(t_data, U_data, 'o', color='orange', label='Messdaten')
plt.xlabel('t [s]')
plt.ylabel('U(t) [V]')
plt.title('Ladekurve eines Kondensators')
plt.legend()
plt.grid(True)
plt.show()

# Automatische Schätzung der Anfangswerte
A0 = np.min(U_data)                    # A0 = niedrigste Spannung
Q0 = np.max(U_data)                    # Q0 = höchste Spannung
U_63 = A0 + 0.63 * (Q0 - A0)           # 63%-Punkt
tau_index = np.argmin(np.abs(U_data - U_63))
tau0 = t_data[tau_index]              # tau = Zeit an diesem Punkt

initial_guess = [A0, Q0, tau0]

# Modellfunktion
def model_function(t, A, Q, tau):
    return A + (Q - A) * (1 - np.exp(-t / tau))

# Residuenfunktion für den Fit
def residuals(params, t, U):
    A, Q, tau = params
    return model_function(t, A, Q, tau) - U

# Nichtlinearer Fit mit gedämpftem Gauss-Newton-Verfahren
result = least_squares(
    residuals, initial_guess, args=(t_data, U_data),
    xtol=1e-7, method='lm'
)

# Gefittete Parameter extrahieren
A_fit, Q_fit, tau_fit = result.x

# Plot der angepassten Funktion
t_plot = np.arange(0, 3.0, 0.001)
U_fit = model_function(t_plot, A_fit, Q_fit, tau_fit)

plt.figure()
plt.plot(t_data, U_data, 'o', color='orange', label='Messdaten')
plt.plot(t_plot, U_fit, '-', color='orangered', label='Angepasste Funktion')
plt.xlabel('t [s]')
plt.ylabel('U(t) [V]')
plt.title('Nichtlineare Ausgleichsrechnung mit Gauss-Newton')
plt.legend()
plt.grid(True)
plt.show()

# Ausgabe der gefitteten Parameter
print("b) Gefittete Parameter:")
print(f"(A, Q, τ) = ({A_fit:.8f}, {Q_fit:.8f}, {tau_fit:.8f})")
print("a) Geschätzte Parameter:")
print(f"(A0, Q0, τ0) = ({initial_guess[0]:.8f}, {initial_guess[1]:.8f}, {initial_guess[2]:.8f})")
