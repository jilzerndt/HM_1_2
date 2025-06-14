import numpy as np
import pandas as pd

def Name_S9_Aufg3(f, a, b, m):
    T = np.zeros((m+1, m+1))

    for j in range(m+1):
        n_j = 2**j
        h_j = (b - a) / n_j
        x = np.linspace(a, b, n_j + 1)
        y = f(x)
        T[j, 0] = h_j * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])  # T_{j0}

    # Romberg-Extrapolation
    for k in range(1, m+1):
        for j in range(k, m+1):
            T[j, k] = (4**k * T[j, k-1] - T[j-1, k-1]) / (4**k - 1)

    return T[m, m], T  # genauester Wert und gesamte Tabelle

# Serie 9, Aufg 2:f(x) = cos(x^2), a = 0, b = pi, m = 4
f = lambda x: np.cos(x**2)
a = 0
b = np.pi
m = 4

exact_value, table = Name_S9_Aufg3(f, a, b, m)
romberg_df = pd.DataFrame(table, columns=[f'k={k}' for k in range(m+1)],
                          index=[f'j={j}' for j in range(m+1)])

print("Romberg-Ergebnisse:")
print(romberg_df)

exact_value