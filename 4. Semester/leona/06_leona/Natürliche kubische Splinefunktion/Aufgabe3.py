import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
population = np.array([75.995, 91.972, 105.711, 123.203, 131.669, 150.697, 179.323, 203.212, 226.505, 249.633, 281.422, 308.745])

# a) Originaldaten plotten
plt.figure(figsize=(10, 6))
plt.scatter(years, population, color='red', label='Gegebene Daten')

# b) Kubische Spline-Interpolation
spline = interpolate.CubicSpline(years, population)
years_fine = np.linspace(1900, 2010, 500)
pop_spline = spline(years_fine)
plt.plot(years_fine, pop_spline, label='Kubische Spline-Interpolation', linestyle='dashed')

# c) Polynominterpolation 11. Grades
years_shifted = years - 1900  # Verschieben der Jahre, um numerische Stabilität zu gewährleisten
coeffs = np.polyfit(years_shifted, population, 11)
poly_values = np.polyval(coeffs, years_fine - 1900)
plt.plot(years_fine, poly_values, label='Polynominterpolation (11. Grad)', linestyle='dotted')


plt.xlabel('Jahr')
plt.ylabel('Bevölkerung (Mio.)')
plt.title('Interpolation der US-Bevölkerungsdaten')
plt.legend()
plt.grid()
plt.show()