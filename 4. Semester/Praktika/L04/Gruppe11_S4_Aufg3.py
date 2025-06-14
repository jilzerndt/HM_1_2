import numpy as np
import matplotlib.pyplot as plt

x_years = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010])
y_ownership = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7])

poly_coeffs = np.polyfit(x_years, y_ownership, len(x_years)-1)
x_plot = np.arange(1975, 2021, 0.1)
y_plot = np.polyval(poly_coeffs, x_plot)

plt.scatter(x_years, y_ownership, color='red', label='Data')
plt.plot(x_plot, y_plot, label='Polynomial Fit', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Computer Ownership (%)')
plt.legend()
plt.show()

# Centered approach
x_centered = x_years - x_years.mean()
centered_coeffs = np.polyfit(x_centered, y_ownership, len(x_years)-1)
y_centered_plot = np.polyval(centered_coeffs, x_plot - x_years.mean())

plt.figure(figsize=(8, 5))
plt.scatter(x_years, y_ownership, color='red', label='Data')
plt.plot(x_plot, y_centered_plot, label='Centered Polynomial Fit', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Computer Ownership (%)')
plt.legend()
plt.show()

# Estimate for 2020
estimate_2020 = np.polyval(centered_coeffs, 2020 - x_years.mean())
print(f"Estimated ownership in 2020: {estimate_2020:.2f}%")
