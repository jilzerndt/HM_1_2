import numpy as np


# === Funktion definieren ===
def f(x):
    return 6 * x ** 2 - 2 * x  # <- Hier kannst du deine eigene Funktion eintragen


# === Parameter für die Integration ===
a = 0  # Untere Grenze (anpassen bei anderer Aufgabe)
b = 4  # Obere Grenze
n_values = [1, 4]  # n entspricht (b-a)/h, also n=1 (h=4) und n=4 (h=1)
# Alternativ kannst du direkt mit Schrittweiten arbeiten und dann n berechnen

for n in n_values:
    h = (b - a) / n
    assert n % 2 == 0 or n == 1, "Simpson-Regel benötigt gerades n; n=1 ergibt direkt ein Intervall"

    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Summierte Simpson-Regel:
    # S = h/3 * (y0 + 4 * (ungerade Indizes) + 2 * (gerade Indizes, außer erste/letzte) + yn)
    weights = np.zeros_like(y)
    weights[0] = 1
    weights[-1] = 1
    for i in range(1, n):
        weights[i] = 4 if i % 2 == 1 else 2

    S = (h / 3) * np.dot(weights, y)
    print(f"n = {n:2d} (h = {h:.2f}): Simpson-Näherung S ≈ {S:.6f}")
