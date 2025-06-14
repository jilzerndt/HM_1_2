import numpy as np
import matplotlib.pyplot as plt

def Gruppe11_S11_Aufg3(f, a, b, n, y0):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    # Initialisiere Lösungsarrays
    y_euler = np.zeros(n + 1)
    y_mittelpunkt = np.zeros(n + 1)
    y_modeuler = np.zeros(n + 1)
    
    y_euler[0] = y0
    y_mittelpunkt[0] = y0
    y_modeuler[0] = y0
    
    print("Starte numerische Integration:")
    print(f"{'i':<3} | {'xi':<6} | {'Euler':<8} | {'Mittelpunkt':<11} | {'Mod. Euler':<11}")
    print("-"*50)

    for i in range(n):
        xi = x[i]
        yi_e = y_euler[i]
        yi_m = y_mittelpunkt[i]
        yi_me = y_modeuler[i]

        # --- Euler ---
        k1_e = f(xi, yi_e)
        y_euler[i + 1] = yi_e + h * k1_e

        # --- Mittelpunkt ---
        k1_m = f(xi, yi_m)
        y_half = yi_m + 0.5 * h * k1_m
        x_half = xi + 0.5 * h
        k2_m = f(x_half, y_half)
        y_mittelpunkt[i + 1] = yi_m + h * k2_m

        # --- Modifiziertes Euler ---
        k1_me = f(xi, yi_me)
        y_pred = yi_me + h * k1_me
        k2_me = f(xi + h, y_pred)
        y_modeuler[i + 1] = yi_me + 0.5 * h * (k1_me + k2_me)

        # --- Ausgabe ---
        print(f"{i:<3} | {xi:.2f} → {x[i+1]:.2f} | {y_euler[i+1]:.5f} | {y_mittelpunkt[i+1]:.5f} | {y_modeuler[i+1]:.5f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_euler, label='Euler', linewidth=2)
    plt.plot(x, y_mittelpunkt, label='Mittelpunkt', linewidth=2)
    plt.plot(x, y_modeuler, label='Mod. Euler', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.show()

    return x, y_euler, y_mittelpunkt, y_modeuler

def f(x, y):
    return x**2 / y

x, y_euler, y_mitte, y_mod = Gruppe11_S11_Aufg3(f, 0, 1.4, 2, 2)

