import numpy as np
import matplotlib.pyplot as plt

# === Anfangsbedingungen ===
y1_0 = 3   # y(1)
y2_0 = 5   # y'(1)
y3_0 = 8   # y''(1)
x_start = 1
x_end = 4

# === Rechte Seite der DGL ===
def rhs(x, y1, y2, y3):
    dy1 = y2
    dy2 = y3
    dy3 = (6*x**4 + 3*x**2*y3 - 6*x*y2 + 6*y1) / x**3
    return np.array([dy1, dy2, dy3])

# === Exakte Lösung zum Vergleich ===
def exact_solution(x):
    return 2*x + x**2 - x**3 + x**4

# === Mittelpunktverfahren ===
def midpoint_method(h):
    N = int((x_end - x_start) / h)
    x_vals = np.linspace(x_start, x_end, N + 1)
    y = np.zeros((N + 1, 3))  # Spalten: y1, y2, y3
    y[0, :] = [y1_0, y2_0, y3_0]

    for i in range(N):
        xi = x_vals[i]
        yi = y[i, :]

        k1 = rhs(xi, *yi)
        y_half = yi + 0.5 * h * k1
        x_half = xi + 0.5 * h
        k2 = rhs(x_half, *y_half)

        y[i + 1, :] = yi + h * k2

    return x_vals, y[:, 0]  # Rückgabe: x-Werte und y1(x) = y(x)

# === Integration für zwei Schrittweiten ===
h_list = [0.2, 0.02]
solutions = []

for h in h_list:
    x_vals, y_approx = midpoint_method(h)
    y_exact = exact_solution(x_vals)
    solutions.append((x_vals, y_approx, y_exact))

    # === Plot numerische vs. exakte Lösung ===
    plt.plot(x_vals, y_approx, label=f'h = {h}', linestyle='--')
    plt.plot(x_vals, y_exact, label='Exakt', linestyle='-')

plt.title('Numerische Lösung vs. exakte Lösung')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.grid(True)
plt.show()

# === Fehlerplot: |y_num - y_exact| (halblogarithmisch) ===
plt.figure()
for x_vals, y_num, y_ex in solutions:
    error = np.abs(y_num - y_ex)
    plt.semilogy(x_vals, error, label=f'h = {x_vals[1]-x_vals[0]}')

plt.title('Fehler |y_numerisch - y_exakt|')
plt.xlabel('x')
plt.ylabel('Fehler (log)')
plt.grid(True, which='both')
plt.legend()
plt.show()
