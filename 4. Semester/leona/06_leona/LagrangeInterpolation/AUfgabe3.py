# kryeziu_S3_Aufg3.py

import numpy as np
import matplotlib.pyplot as plt

# Originaldaten
years = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010])
percent = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7])

# --- Aufgabe a: polyfit mit Originaldaten ---
# Grad = Anzahl Punkte - 1
degree = len(years) - 1
coeffs_a = np.polyfit(years, percent, degree)
x_plot = np.arange(1975, 2020.1, 0.1)
y_plot_a = np.polyval(coeffs_a, x_plot)

# --- Aufgabe b: x zentriert um Mittelwert ---
x_mean = years.mean()
years_centered = years - x_mean
x_plot_centered = x_plot - x_mean
coeffs_b = np.polyfit(years_centered, percent, degree)
y_plot_b = np.polyval(coeffs_b, x_plot_centered)

# --- Plot erstellen ---
plt.figure(figsize=(10, 6))
plt.plot(years, percent, 'b+', label='data')
plt.plot(x_plot, y_plot_a, 'orange', label='polyfit')
plt.plot(x_plot, y_plot_b, 'green', label='polyfit für x-x.mean()')
plt.xlabel("Jahr")
plt.ylabel("Haushalte mit Computer [%]")
plt.ylim([-100, 250])
plt.xlim([1975, 2020])
plt.grid(True)
plt.legend()
plt.title("Polynomial Fit für Computerbesitz in Haushalten (USA)")
plt.show()

# --- Kommentar als Antwort auf Teil a ---
# Obwohl der Fit aus Aufgabe a den Grad n = 9 hat, geht er NICHT exakt durch alle Punkte.
# Der Grund ist numerische Instabilität bei hohen Graden über große x-Werte.
# Der Fit aus Teil b mit x - mean(x) vermeidet das Problem – der grüne Fit geht exakt durch alle Punkte.
