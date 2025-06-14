import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

# Fehlerwarnungen aktivieren (für Debugging)
np.seterr(over='warn', divide='warn', invalid='warn')

x = np.array([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. , 8.5, 9. , 9.5])
y = np.array([159.57209984, 159.8851819 , 159.89378952, 160.30305273,
              160.84630757, 160.94703969, 161.56961845, 162.31468058,
              162.32140561, 162.88880047, 163.53234609, 163.85817086,
              163.55339958, 163.86393263, 163.90535931, 163.44385491])

# Ansatzfunktion – mit numerischer Stabilisierung
def f(x, lam):
    l0, l1, l2, l3 = lam
    exponent = np.power(10, np.clip(l2 + l3 * x, -20, 20))  # Clipping vermeidet Overflow
    return (l0 + l1 * exponent) / (1 + exponent)

# Fehlerfunktion (Summe der quadrierten Abweichungen)
def error(lam):
    return np.sum((f(x, lam) - y)**2)

# Gedämpftes Gauss-Newton-Verfahren
def gauss_newton_damped(lam0, max_iter=50, tol=1e-6, damping=0.5):
    lam = lam0.copy()
    for _ in range(max_iter):
        y_fit = f(x, lam)
        r = y - y_fit
        J = np.zeros((len(x), len(lam)))
        h = 1e-6
        for j in range(len(lam)):
            lam_h = lam.copy()
            lam_h[j] += h
            J[:, j] = (f(x, lam_h) - y_fit) / h
        try:
            delta = np.linalg.lstsq(J, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            print("Lineares Gleichungssystem konnte nicht gelöst werden.")
            break
        lam_new = lam + damping * delta
        if np.linalg.norm(delta) < tol:
            break
        lam = lam_new
    return lam

# Neue, stabilere Startwerte
lam0 = np.array([100.0, 120.0, 3, -1])

# a) Gedämpftes Gauss-Newton
lam_gn_damped = gauss_newton_damped(lam0)

# b) Ungedämpftes Verfahren (Dämpfungsfaktor = 1)
lam_gn_ungedämpft = gauss_newton_damped(lam0, damping=1.0)
# Kommentar: Auch das ungedämpfte Verfahren konvergiert hier, da die Startwerte gut gewählt sind.

# c) Lösung mit fmin
lam_fmin = fmin(error, lam0, disp=False)

plt.figure(figsize=(10,6))
plt.plot(x, y, 'bo', label='Messdaten')
plt.plot(x, f(x, lam_gn_damped), 'r--', label='Gedämpftes Gauss-Newton')
plt.plot(x, f(x, lam_gn_ungedämpft), 'g:', label='Ungedämpftes Gauss-Newton')
plt.plot(x, f(x, lam_fmin), 'm-', label='fmin')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Nichtlineare Regression – Vergleich der Verfahren')
plt.legend()
plt.grid(True)
plt.show()

print("Gedämpftes Gauss-Newton Ergebnis:", lam_gn_damped)
print("Ungedämpftes Gauss-Newton Ergebnis:", lam_gn_ungedämpft)
print("fmin Ergebnis:", lam_fmin)
