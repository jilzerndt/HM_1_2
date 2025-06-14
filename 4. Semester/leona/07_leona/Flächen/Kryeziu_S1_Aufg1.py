import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definitionsbereiche und Funktionen

def wurfweite(v0, alpha):
    g = 9.81  # Erdbeschleunigung in m/s^2
    return (v0**2 * np.sin(2 * np.radians(alpha))) / g

v0 = np.linspace(0, 100, 50)
alpha = np.linspace(0, 90, 50)
V, A = np.meshgrid(v0, alpha)
W = wurfweite(V, A)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(V, A, W)
ax.set_xlabel("Anfangsgeschwindigkeit v0 (m/s)")
ax.set_ylabel("Winkel alpha (Grad)")
ax.set_zlabel("Wurfweite W (m)")
ax.set_title("Wurfweite als Funktion von v0 und alpha (Wireframe)")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(V, A, W, cmap='viridis')
fig.colorbar(surf, label="Wurfweite W (m)")
ax.set_xlabel("Anfangsgeschwindigkeit v0 (m/s)")
ax.set_ylabel("Winkel alpha (Grad)")
ax.set_zlabel("Wurfweite W (m)")
ax.set_title("Wurfweite als Funktion von v0 und alpha (Surface)")
plt.show()

plt.figure()
plt.contourf(V, A, W, cmap='viridis')
plt.colorbar(label="Wurfweite W (m)")
plt.xlabel("Anfangsgeschwindigkeit v0 (m/s)")
plt.ylabel("Winkel alpha (Grad)")
plt.title("Konturplot der Wurfweite")
plt.show()

# Funktionen für das ideale Gasgesetz
def p_vt(V, T):
    R = 8.31  # Gaskonstante in J/(mol*K)
    return (R * T) / V

def v_pt(p, T):
    R = 8.31
    return (R * T) / p

def t_pv(p, V):
    R = 8.31
    return (p * V) / R

# Definitionsbereiche
V = np.linspace(0.01, 0.2, 50)
T = np.linspace(1, 1e4, 50)
P = np.linspace(1e4, 1e5, 50)
P2 = np.linspace(1e4, 1e6, 50)
V2 = np.linspace(0.01, 10, 50)

V, T = np.meshgrid(V, T)
P_grid, T_grid = np.meshgrid(P, T)
P2_grid, V2_grid = np.meshgrid(P2, V2)

P_values = p_vt(V, T)
V_values = v_pt(P_grid, T_grid)
T_values = t_pv(P2_grid, V2_grid)

# Plots für das ideale Gasgesetz
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(V, T, P_values)
ax.set_xlabel("Volumen V (m^3)")
ax.set_ylabel("Temperatur T (K)")
ax.set_zlabel("Druck p (N/m^2)")
ax.set_title("Druck p als Funktion von V und T (Wireframe)")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(V, T, P_values, cmap='plasma')
fig.colorbar(surf, label="Druck p (N/m^2)")
ax.set_xlabel("Volumen V (m^3)")
ax.set_ylabel("Temperatur T (K)")
ax.set_zlabel("Druck p (N/m^2)")
ax.set_title("Druck p als Funktion von V und T (Surface)")
plt.show()

plt.figure()
plt.contourf(V, T, P_values, cmap='coolwarm')
plt.colorbar(label="Druck p (N/m^2)")
plt.xlabel("Volumen V (m^3)")
plt.ylabel("Temperatur T (K)")
plt.title("Konturplot des Drucks")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(P_grid, T_grid, V_values)
ax.set_xlabel("Druck p (N/m^2)")
ax.set_ylabel("Temperatur T (K)")
ax.set_zlabel("Volumen V (m^3)")
ax.set_title("Volumen V als Funktion von p und T (Wireframe)")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P_grid, T_grid, V_values, cmap='plasma')
fig.colorbar(surf, label="Volumen V (m^3)")
ax.set_xlabel("Druck p (N/m^2)")
ax.set_ylabel("Temperatur T (K)")
ax.set_zlabel("Volumen V (m^3)")
ax.set_title("Volumen V als Funktion von p und T (Surface)")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(P2_grid, V2_grid, T_values)
ax.set_xlabel("Druck p (N/m^2)")
ax.set_ylabel("Volumen V (m^3)")
ax.set_zlabel("Temperatur T (K)")
ax.set_title("Temperatur T als Funktion von p und V (Wireframe)")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P2_grid, V2_grid, T_values, cmap='plasma')
fig.colorbar(surf, label="Temperatur T (K)")
ax.set_xlabel("Druck p (N/m^2)")
ax.set_ylabel("Volumen V (m^3)")
ax.set_zlabel("Temperatur T (K)")
ax.set_title("Temperatur T als Funktion von p und V (Surface)")
plt.show()
