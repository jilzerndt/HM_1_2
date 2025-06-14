import numpy as np

# Daten
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0.54, 0.44, 0.28, 0.18, 0.12, 0.08])

# Vandermonde-Matrix (höchster Grad zuerst)
A = np.vander(x, 5)  # ergibt x^4 bis x^0

# Least Squares Lösung
coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)

# Ausgabe
print(np.round(coeffs, 8))  # ergibt genau deine Koeffizienten
