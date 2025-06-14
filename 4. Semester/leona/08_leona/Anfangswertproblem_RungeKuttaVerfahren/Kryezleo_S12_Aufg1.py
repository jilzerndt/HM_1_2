import numpy as np
import pandas as pd

def Leona_S12_Aufg1(f, a, b, n, y0):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    # Für die Tabelle
    table = {
        "i": [],
        "t_i": [],
        "y_i": [],
        "k1": [],
        "k2": [],
        "k3": [],
        "k4": [],
        "y_{i+1}": []
    }

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Tabelle füllen
        table["i"].append(i)
        table["t_i"].append(round(x[i], 4))
        table["y_i"].append(round(y[i], 4))
        table["k1"].append(round(k1, 4))
        table["k2"].append(round(k2, 4))
        table["k3"].append(round(k3, 4))
        table["k4"].append(round(k4, 4))
        table["y_{i+1}"].append(round(y[i+1], 4))

    # Letzter Punkt (nur für Übersichtlichkeit in Tabelle)
    table["i"].append(n)
    table["t_i"].append(round(x[n], 4))
    table["y_i"].append(round(y[n], 4))
    table["k1"].append("-")
    table["k2"].append("-")
    table["k3"].append("-")
    table["k4"].append("-")
    table["y_{i+1}"].append("-")

    df = pd.DataFrame(table)
    return x, y, df

# Beispiel 8.7 Funktion
def f_example(t, y):
    return t**2 + 0.1 * y

# Parameter
a = -1.5
b = 1.5
n = 5
y0 = 0

# Anwendung
x_vals, y_vals, tabelle = Leona_S12_Aufg1(f_example, a, b, n, y0)

# Tabelle anzeigen
print(tabelle)