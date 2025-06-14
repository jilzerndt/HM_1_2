import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def Name_S5_Aufg2(x, y, xx):
    # Erstelle eine natürliche kubische Spline-Interpolation
    cs = CubicSpline(x, y, bc_type='natural')

    # Berechne die interpolierten Werte
    yy = cs(xx)

    # Grafische Darstellung
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'o', label='Stützpunkte')  # Originalpunkte
    plt.plot(xx, yy, '-', label='Kubischer Spline')  # Interpolierte Kurve
    plt.xlabel('x')
    plt.ylabel('S(x)')
    plt.legend()
    plt.title('Natürliche kubische Spline-Interpolation')
    plt.grid()
    plt.show()

    # Ausgabe der Splineterme für jedes Intervall
    for i in range(len(x) - 1):
        print(
            f"S{i}(x) = {cs.c[3, i]:.4f} * (x - {x[i]})^3 + {cs.c[2, i]:.4f} * (x - {x[i]})^2 + {cs.c[1, i]:.4f} * (x - {x[i]}) + {cs.c[0, i]:.4f}")

    return yy


# Beispielwerte gemäß Aufgabe 1
x = np.array([4, 6, 8, 10])
y = np.array([6, 3, 9, 0])
xx = np.linspace(4, 10, 100)  # Feinere Werte für die Darstellung

# Funktion aufrufen
yy = Name_S5_Aufg2(x, y, xx)

# Die lösung hier ist nicht gleich wie bei der Aufgabe eins, die exponenten bei der Aufgabe 1 fängt links mit 1 an und geht richtung rechts bis max hoch 3