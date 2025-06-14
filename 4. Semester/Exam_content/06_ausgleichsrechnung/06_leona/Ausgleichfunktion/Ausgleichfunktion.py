import numpy as np
import matplotlib.pyplot as plt

# ================================================
# 🔁 1. DATEN ANPASSEN: Zeit- und Messwerte
# Wenn die Aufgabe andere Werte oder andere Anzahl Punkte hat,
# dann hier t (Zeitpunkte) und p (Bevölkerungszahlen o.ä.) ändern.
# ================================================
t = np.arange(0, 111, 10)  # Zeitwerte (z.B. alle 10 Jahre ab 1900)
p = np.array([76, 92, 106, 123, 137, 151, 179, 203, 227, 250, 281, 309])  # Messwerte

# ================================================
# 🧮 2. FUNKTIONSGRAD ANPASSEN falls andere in der aufgabenstellung:
# Willst du andere Funktionen (z.B. nur Grad 2 oder höherer Grad),
# dann hier Designmatrizen anpassen.
# ================================================

# Designmatrix für Polynom 3. Grades: p₁(t) = a₃t³ + a₂t² + a₁t + a₀
A1 = np.vstack([t**3, t**2, t, np.ones_like(t)]).T

# Designmatrix für Polynom 2. Grades: p₂(t) = b₂t² + b₁t + b₀
A2 = np.vstack([t**2, t, np.ones_like(t)]).T

# ================================================
# 🔍 3. KOEFFIZIENTEN BESTIMMEN
# Kein Anpassen nötig, falls du Least-Squares verwenden willst.
# ================================================
a = np.linalg.lstsq(A1, p, rcond=None)[0]  # p₁(t)-Koeffizienten
b = np.linalg.lstsq(A2, p, rcond=None)[0]  # p₂(t)-Koeffizienten

# ================================================
# 📈 4. GRAFIKBEREICH ANPASSEN
# Falls andere Zeitintervalle, dann t_fine Bereich anpassen.
# ================================================
t_fine = np.linspace(min(t), max(t), 500)

# Berechnung der Ausgleichskurven
p1_fit = a[0]*t_fine**3 + a[1]*t_fine**2 + a[2]*t_fine + a[3]
p2_fit = b[0]*t_fine**2 + b[1]*t_fine + b[2]

# Berechnung der approximierten Werte an den Originalpunkten
p1_approx = a[0]*t**3 + a[1]*t**2 + a[2]*t + a[3]
p2_approx = b[0]*t**2 + b[1]*t + b[2]

# ================================================
# 📊 5. PLOT UND DARSTELLUNG
# Text anpassen, wenn du andere Funktionen oder Daten darstellst.
# ================================================
plt.figure(figsize=(10, 6))
plt.plot(t, p, 'ko', label='Originaldaten')
plt.plot(t_fine, p1_fit, 'b-', label='p₁(t) = a₃t³ + a₂t² + a₁t + a₀') # Anpassen je nach Funktion
plt.plot(t_fine, p2_fit, 'r--', label='p₂(t) = b₂t² + b₁t + b₀') # Anpassen je nach Funktion
plt.xlabel('t [Jahre seit 1900]')  # Anpassen, falls t etwas anderes ist
plt.ylabel('Bevölkerung p(t) [Mio.]')  # Anpassen je nach Messgröße
plt.title('Ausgleichsfunktionen für Messdaten')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================
# ✅ 6. FEHLERFUNKTIONALE
# Bleibt gleich – vergleicht die Güte der Anpassung.
# ================================================
error1 = np.sum((p - p1_approx)**2)
error2 = np.sum((p - p2_approx)**2)

print(f"Fehlerfunktional für p₁(t): {error1:.2f}")
print(f"Fehlerfunktional für p₂(t): {error2:.2f}")

if error1 < error2:
    print("✅ p₁(t) liefert die bessere Näherung an die Daten.")
else:
    print("✅ p₂(t) liefert die bessere Näherung.")
