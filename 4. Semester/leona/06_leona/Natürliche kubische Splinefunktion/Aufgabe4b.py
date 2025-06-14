import sympy as sp

# === Symbole definieren (anpassen für andere Aufgaben) ===
a, b, c, d, x = sp.symbols('a b c d x')

# === Spline-Stücke definieren (hier konkret aus Aufgabe) ===
S0 = a + 2*x + c*x**2 + 0.5*x**3                # Für x ∈ [0, 2]
S1 = 6 + b*(x - 2) + 2*(x - 2)**2 + d*(x - 2)**3 # Für x ∈ [2, 3]

# === 0. bis 3. Ableitung berechnen ===
S0_1 = sp.diff(S0, x)
S1_1 = sp.diff(S1, x)

S0_2 = sp.diff(S0_1, x)
S1_2 = sp.diff(S1_1, x)

S0_3 = sp.diff(S0_2, x)
S1_3 = sp.diff(S1_2, x)

# === Bedingungen für Spline-Gleichheit an x = 2 ===
# 1. Stetigkeit der Funktion
eq1 = sp.Eq(S0.subs(x, 2), S1.subs(x, 2))

# 2. Stetigkeit der ersten Ableitung
eq2 = sp.Eq(S0_1.subs(x, 2), S1_1.subs(x, 2))

# 3. Stetigkeit der zweiten Ableitung
eq3 = sp.Eq(S0_2.subs(x, 2), S1_2.subs(x, 2))

# 4. Not-a-knot-Bedingung: dritte Ableitungen gleich
eq4 = sp.Eq(S0_3.subs(x, 2), S1_3.subs(x, 2))

# === Gleichungssystem lösen ===
lösungen = sp.solve([eq1, eq2, eq3, eq4], (a, b, c, d))

# === Ergebnisse anzeigen ===
print("Lösungen für a, b, c, d:")
for var, val in lösungen.items():
    print(f"{var} = {val}")
