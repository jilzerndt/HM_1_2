import numpy as np
import matplotlib.pyplot as plt

T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
rho = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4])

A = np.vstack([T**2, T, np.ones_like(T)]).T

ATA = A.T @ A
ATy = A.T @ rho
coeffs_normal = np.linalg.solve(ATA, ATy)
a1, b1, c1 = coeffs_normal

Q, R = np.linalg.qr(A)
coeffs_qr = np.linalg.solve(R, Q.T @ rho)
a2, b2, c2 = coeffs_qr

T_plot = np.linspace(0, 100, 500)
f1 = a1*T_plot**2 + b1*T_plot + c1
f2 = a2*T_plot**2 + b2*T_plot + c2

plt.plot(T, rho, 'o', label='Messdaten')
plt.plot(T_plot, f1, '-', label='Normalgleichung')
plt.plot(T_plot, f2, '--', label='QR-Zerlegung')


coeffs_polyfit = np.polyfit(T, rho, 2)
a3, b3, c3 = coeffs_polyfit
f3 = a3*T_plot**2 + b3*T_plot + c3
plt.plot(T_plot, f3, ':', label='numpy.polyfit()')

plt.xlabel('Temperatur T [°C]')
plt.ylabel('Dichte ρ [g/l]')
plt.legend()
plt.grid(True)
plt.title('Ausgleichsfunktion der Dichte von Wasser')
plt.show()


cond_ATA = np.linalg.cond(ATA)
cond_R = np.linalg.cond(R)
print("Konditionszahl ATA:", cond_ATA)
print("Konditionszahl R:", cond_R)


def fehlerfunktional(T, rho, coeffs):
    f = coeffs[0]*T**2 + coeffs[1]*T + coeffs[2]
    return np.sum((rho - f)**2)

err_normal = fehlerfunktional(T, rho, coeffs_normal)
err_polyfit = fehlerfunktional(T, rho, coeffs_polyfit)

print("Fehlerfunktional (Normalgleichung):", err_normal)
print("Fehlerfunktional (numpy.polyfit):", err_polyfit)
