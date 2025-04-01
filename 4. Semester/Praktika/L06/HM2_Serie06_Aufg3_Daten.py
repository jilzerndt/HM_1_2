# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:03:41 2021

Höhere Mathematik 2, Serie 6, Aufgabe 3, Daten

@author: knaa
"""

import numpy as np
import matplotlib.pyplot as plt

data=np.array([
    [1971, 2250.],
    [1972, 2500.],
    [1974, 5000.],
    [1978, 29000.],
    [1982, 120000.],
    [1985, 275000.],
    [1989, 1180000.],
    [1989, 1180000.],
    [1993, 3100000.],
    [1997, 7500000.],
    [1999, 24000000.],
    [2000, 42000000.],
    [2002, 220000000.],
    [2003, 410000000.],   
    ])

years = data[:, 0]
transistor_counts = data[:, 1]

log_transistor_counts = np.log10(transistor_counts)

X_log = np.vstack((years - 1970, np.ones(len(years)))).T

# Normal equation
theta_log = np.linalg.inv(X_log.T @ X_log) @ X_log.T @ log_transistor_counts

year_2015 = 2015
log_N_2015 = theta_log[0] * (year_2015 - 1970) + theta_log[1]
N_2015 = 10 ** log_N_2015

print("Computed coefficients (theta values):", theta_log)
print(f"log10(N) = {theta_log[0]:.4f} * (t - 1970) + {theta_log[1]:.4f}")
print(f"Predicted transistor count for 2015: {N_2015:.2e}")

# Compute predicted log-transistor counts
log_transistor_pred = X_log @ theta_log

plt.figure(figsize=(8, 5))
plt.scatter(years, log_transistor_counts, color='blue', label="Actual Data (log scale)")
plt.plot(years, log_transistor_pred, color='red', linestyle='--', label="Fitted Line")
plt.xlabel("Year")
plt.ylabel("log10(Number of Transistors)")
plt.title("Logarithmic Fit for Transistor Growth (Exercise 3)")
plt.legend()
plt.grid(True)
plt.show()
