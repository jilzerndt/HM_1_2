def lagrange_int(x, y, x_int):
    """
    Berechnet die Lagrange-Interpolation für den gegebenen Wert x_int.
    :param x: Liste der x-Werte (Höhenwerte)
    :param y: Liste der y-Werte (Atmosphärendruckwerte)
    :param x_int: Der Punkt, für den der y-Wert interpoliert wird
    :return: Interpolierter y-Wert
    """
    n = len(x)
    y_int = 0

    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_int - x[j]) / (x[i] - x[j])
        y_int += term

    return y_int


# Gegebene Daten
x_vals = [0, 2500, 5000, 10000]
y_vals = [1013, 747, 540, 226]
x_missing = 3750

# Interpolation des fehlenden Wertes
y_interpolated = lagrange_int(x_vals, y_vals, x_missing)
print(f"Der interpolierte Wert für 3750 m Höhe ist: {y_interpolated:.2f} hPa")
