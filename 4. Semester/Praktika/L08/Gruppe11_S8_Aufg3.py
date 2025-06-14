import numpy as np
import matplotlib.pyplot as plt

def trapez_neq(x, y):
    """
    x: Liste oder Array der x-Werte
    y: Liste oder Array der y-Werte
    """
    if len(x) != len(y):
        raise ValueError("x und y müssen gleich lang sein.")
    
    integral = 0.0
    for i in range(len(x) - 1):
        dx = x[i+1] - x[i]
        avg = (y[i] + y[i+1]) / 2
        integral += avg * dx
    return integral


# Daten aus aufgabenstellung
r_km = np.array([0, 800, 1200, 1400, 2000, 3000, 3400, 3600, 4000, 5000, 5500, 6370])
rho = np.array([13000, 12900, 12700, 12000, 11650, 10600, 9900, 5500, 5300, 4750, 4500, 3300])

r_m = r_km * 1000  # Radius in Meter

# integrand
f_r = rho * 4 * np.pi * r_m**2

masse_erde = trapez_neq(r_m, f_r)

# Tatsächliche erdmasse
masse_erde_literatur = 5.972e24 

absoluter_fehler = abs(masse_erde - masse_erde_literatur)
relativer_fehler = absoluter_fehler / masse_erde_literatur

print(f"{'-'*40}")
print(f"Berechnete Erdmasse: {masse_erde:.4e} kg")
print(f"Literaturwert:       {masse_erde_literatur:.4e} kg")
print(f"Absoluter Fehler:    {absoluter_fehler:.4e} kg")
print(f"Relativer Fehler:    {relativer_fehler*100:.4f} %")
print(f"{'-'*40}")

plt.plot(r_m / 1000, f_r, marker='o')
plt.xlabel('Radius r [km]')
plt.ylabel('Integrand f(r) = ρ·4πr² [kg/m]')
plt.title('Integrand zur Massenberechnung')
plt.grid(True)
plt.show()
