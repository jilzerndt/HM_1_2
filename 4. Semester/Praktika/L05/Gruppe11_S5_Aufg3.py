import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def natural_cubic_spline(x, y, xx):
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n - 1, n - 1))
    rhs = np.zeros(n - 1)

    for i in range(1, n):
        A[i - 1, i - 1] = (h[i - 1] + h[i]) * 2
        if i - 2 >= 0:
            A[i - 1, i - 2] = h[i - 1]
        if i < n - 1:
            A[i - 1, i] = h[i]
        rhs[i - 1] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    m = np.zeros(n + 1)
    if n > 1:
        m[1:n] = np.linalg.solve(A, rhs)

    yy = np.zeros_like(xx)
    for j, x_val in enumerate(xx):
        i = np.searchsorted(x, x_val) - 1
        i = np.clip(i, 0, n - 1)
        dx = x_val - x[i]
        hi = h[i]

        a = (m[i + 1] - m[i]) / (6 * hi)
        b = m[i] / 2
        c = (y[i + 1] - y[i]) / hi - (2 * hi * m[i] + hi * m[i + 1]) / 6
        d = y[i]

        yy[j] = a * dx**3 + b * dx**2 + c * dx + d

    return yy

# Befölkerungdaten USA
t = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
p = np.array([75.995, 91.972, 105.711, 123.203, 131.669, 150.697,
              179.323, 203.212, 226.505, 249.633, 281.422, 308.745])

tt = np.linspace(t[0], t[-1], 500)

# Eigene spline
yy_own = natural_cubic_spline(t, p, tt)

# Scipy
cs = interpolate.CubicSpline(t, p, bc_type='natural')
yy_scipy = cs(tt)

# Polynom 11. Grades
t_shifted = t - 1900
coeffs = np.polyfit(t_shifted, p, 11)
tt_shifted = tt - 1900
yy_poly = np.polyval(coeffs, tt_shifted)

plt.figure(figsize=(10, 6))
plt.plot(t, p, 'o', label='Daten')
plt.plot(tt, yy_own, label='Eigener Spline', linestyle='--')
plt.plot(tt, yy_scipy, label='Scipy Spline', linestyle='-.')
plt.plot(tt, yy_poly, label='Polynom 11. Grades', linestyle=':')
plt.xlabel('Jahr')
plt.ylabel('Bevölkerung (Mio.)')
plt.title('Interpolation der US-Bevölkerungszahlen')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
