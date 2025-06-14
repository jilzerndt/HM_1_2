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

t = data[:, 0]               # Jahre
N = data[:, 1]               # Transistoranzahl
logN = np.log10(N)           # Logarithmierte Anzahl

A = np.vstack([np.ones_like(t), t - 1970]).T
theta = np.linalg.lstsq(A, logN, rcond=None)[0]
theta1, theta2 = theta

t_plot = np.linspace(1970, 2010, 500)
logN_fit = theta1 + (t_plot - 1970) * theta2
N_fit = 10 ** logN_fit

plt.figure(figsize=(8, 5))
plt.semilogy(t, N, 'o', label="Messdaten")
plt.semilogy(t_plot, N_fit, '-', label="Fit")
plt.xlabel("Jahr")
plt.ylabel("Transistoren pro Chip")
plt.title("Moore'sches Gesetz")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

jahr = 2015
logN_2015 = theta1 + (jahr - 1970) * theta2
N_2015 = 10 ** logN_2015
print(f"Extrapolierte Transistoranzahl im Jahr 2015: {N_2015:.2e} (tatsächlich: ca. 4e9)")

print(f"θ₁ = {theta1:.4f}, θ₂ = {theta2:.4f}")

verdopplungszeit = np.log10(2) / theta2
print(f"Verdopplungszeit (nach Fit): {verdopplungszeit:.2f} Jahre")