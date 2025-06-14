# Kryeziu_S3_Aufg2.py
import sympy as sp

# Symbole definieren
x, y = sp.symbols('x y')

# Funktionen f1 und f2 definieren (Hyperbeln)
f1 = x**2 / 186**2 - y**2 / (300**2 - 186**2) - 1
f2 = (y - 500)**2 / 279**2 - (x - 300)**2 / (500**2 - 279**2) - 1

# Gleichungen implizit plotten
p1 = sp.plot_implicit(sp.Eq(f1, 0), (x, -2000, 2000), (y, -2000, 2000), show=False)
p2 = sp.plot_implicit(sp.Eq(f2, 0), (x, -2000, 2000), (y, -2000, 2000), show=False)

# Zweite Kurve zum Plot hinzufügen
p1.append(p2[0])

# Plot anzeigen
p1.show()
