import numpy as np
from scipy.interpolate import lagrange

def lagrange_int(x, y, x_int):
    """
    Compute Lagrange interpolation.
    :param x: List or numpy array of x values
    :param y: List or numpy array of y values
    :param x_int: The x value(s) at which to interpolate
    :return: Interpolated y value(s)
    """
    poly = lagrange(x, y)
    return poly(x_int)

x_vals = np.array([0, 2500, 3750, 5000, 10000])
y_vals = np.array([1013, 747, np.nan, 540, 226])

y_interpolated = lagrange_int(x_vals[~np.isnan(y_vals)], y_vals[~np.isnan(y_vals)], 3750)
print(f"Interpolated pressure at 3750m: {y_interpolated:.2f} hPa")

