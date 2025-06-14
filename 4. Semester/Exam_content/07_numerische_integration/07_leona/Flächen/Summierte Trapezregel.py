import sympy as sp

# -----------------------------------------------
# ANPASSUNG 1: Definiere hier deine Funktion f(x)
# Beispiel: f(x) = sin(x)
x, h = sp.symbols('x h')
f = sp.sin(x)
# -----------------------------------------------

# Zweite Ableitung von f(x) berechnen
f2 = f.diff(x, 2)
f2_abs = sp.Abs(f2)

# -----------------------------------------------
# ANPASSUNG 2: Passe hier das Intervall [a, b] an
# Beispiel: Integration von 0 bis pi
a = 0
b = sp.pi
# -----------------------------------------------

# -----------------------------------------------
# ANPASSUNG 3: Toleranz für maximalen Fehler setzen
# Beispiel: maximaler Fehler soll 1e-3 sein
toleranz = 1e-3
# -----------------------------------------------

# Kritische Punkte für |f''(x)| im Intervall bestimmen
# (hier: Randpunkte und Extremstellen)
kritische_punkte = [a, b, (a + b) / 2]
f2_max = max([f2_abs.subs(x, punkt).evalf() for punkt in kritische_punkte])

# -----------------------------------------------
# ANPASSUNG 4: Fehlerformel der Trapezregel definieren
# E(h) = (h² / 12)(b - a) * max|f''(x)|
# Wenn du eine andere Formel verwendest (z. B. Simpson),
# musst du diese Zeile entsprechend ändern!
# -----------------------------------------------
error_expr = (h**2 / 12) * (b - a) * f2_max

# Gleichung lösen: Fehler soll gleich Toleranz sein
h_grenze_expr = sp.solve(sp.Eq(error_expr, toleranz), h)
h_grenze_val = [val.evalf() for val in h_grenze_expr if val.is_real and val > 0][0]

# Ausgabe
print("Symbolische Fehlerformel:")
sp.pprint(error_expr)
print(f"\nMaximal erlaubte Schrittweite h: {h_grenze_val:.8f}")
