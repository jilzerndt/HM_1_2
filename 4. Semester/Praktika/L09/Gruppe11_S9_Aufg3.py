import math
def Name_S9_Aufg3(f, a, b, m):
    """
    Romberg-Integration für ∫_a^b f(x) dx
    f: Funktion
    a, b: Integrationsgrenzen
    m: Anzahl Zeilen in der Romberg-Tabelle (j = 0,...,m)
    Gibt letzter Wert der Romberg-Tabelle zurück
    """
    R = [[0] * (i+1) for i in range(m+1)]  # Romberg-Tabelle initialisieren

    for j in range(m+1):
        n = 2**j
        h = (b - a) / n
        summe = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            x = a + i * h
            summe += f(x)
        R[j][0] = h * summe  # Trapezregelwert speichern

    for k in range(1, m+1):
        for j in range(k, m+1):
            R[j][k] = R[j][k-1] + (R[j][k-1] - R[j-1][k-1]) / (4**k - 1)

    return R[m][m]

def integrand(x):
    return math.cos(x**2)

result = Name_S9_Aufg3(integrand, 0, math.pi, 4)
print("Romberg-Ergebnis:", result)
