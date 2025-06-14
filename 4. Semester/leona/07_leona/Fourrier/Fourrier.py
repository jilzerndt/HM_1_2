import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === Parameter ===
T = 2 * np.pi      # Periodendauer (kannst du ändern für andere Aufgaben)
omega = 2 * np.pi / T
n = 10             # Anzahl der Fourier-Terme (hier n = 10)
num_points = 100   # Anzahl der äquidistanten Punkte (kannst du anpassen)

# === Originalfunktion f(t): Hier die gewünschte Funktion definieren ===
def f(t):
    return np.sin(t)  # <- Hier andere Funktion eintragen bei neuer Aufgabe

# === verrauschte Version g(t): Rauschen hinzufügen ===
def g(t):
    noise_strength = 0.3
    return f(t) + noise_strength * np.random.randn(*t.shape)

# === Fourier-Koeffizienten berechnen ===
def compute_fourier_coefficients(g, T, n):
    A0, _ = quad(lambda t: g(t), 0, T)
    A0 = (2 / T) * A0

    Ak = []
    Bk = []
    for k in range(1, n + 1):
        a_k, _ = quad(lambda t: g(t) * np.cos(k * omega * t), 0, T)
        b_k, _ = quad(lambda t: g(t) * np.sin(k * omega * t), 0, T)
        Ak.append((2 / T) * a_k)
        Bk.append((2 / T) * b_k)
    return A0, Ak, Bk

# === Fourier-Näherung h(t) berechnen ===
def fourier_series(t, A0, Ak, Bk):
    result = A0 / 2
    for k in range(1, len(Ak) + 1):
        result += Ak[k - 1] * np.cos(k * omega * t) + Bk[k - 1] * np.sin(k * omega * t)
    return result

# === t-Werte erzeugen ===
t_vals = np.linspace(0, T, num_points)

# === g(t) als verrauschte Daten simulieren ===
g_vals = g(t_vals)

# === Fourier-Koeffizienten berechnen ===
# Hinweis: g für Integration muss skalare Inputs akzeptieren, daher Lambda mit f
g_for_integral = lambda t: f(t) + 0.3 * np.random.randn()
A0, Ak, Bk = compute_fourier_coefficients(g_for_integral, T, n)

# === Fourier-Näherung h(t) berechnen ===
h_vals = fourier_series(t_vals, A0, Ak, Bk)

# === Plot erstellen ===
plt.figure(figsize=(10, 6))
plt.plot(t_vals, f(t_vals), label='f(t)', linewidth=2)
plt.plot(t_vals, g_vals, label='g(t) (verrauscht)', linestyle='--')
plt.plot(t_vals, h_vals, label='h(t) (Fourier-Näherung)', linewidth=2)
plt.title('Fourier-Näherung an verrauschte Funktion g(t)')
plt.xlabel('t')
plt.ylabel('Funktion')
plt.legend()
plt.grid(True)
plt.show()
