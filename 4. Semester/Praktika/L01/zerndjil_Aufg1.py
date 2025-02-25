import numpy as np
import matplotlib.pyplot as plt

'''
jede der Funktionen in a) und b) wie folgt darstellen:

- dreidimensional mit plot_wireframe()
- dreidimensional mit plot_surface() und passender Colormap
- zweidimensional mit den Höhenlinien

Versehen sie jede Abbildung mit passenden Achsenbeschriftungen und einem Titel.
'''

#a) Funktion $W(v_0, \alpha)$ beschreibt die Wurfweite W eines Körpers:
def W(v0, alpha):
    g = 9.81
    return ((v0**2) * np.sin(2*alpha))/g

#Definitionsbereich für alpha:
alpha = np.linspace(0, np.pi/2, 100)
#Definitionsbereich für v0:
v0 = np.linspace(0, 100, 100)

#dreidimensional mit plot_wireframe()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(alpha, v0)
Z = W(Y, X)
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('alpha')
ax.set_ylabel('v0')
ax.set_zlabel('Wurfweite')
ax.set_title('Wurfweite eines Körpers in Abhängigkeit von alpha und v0')
plt.show()

#dreidimensional mit plot_surface() und passender Colormap
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(alpha, v0)
Z = W(Y, X)
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('alpha')
ax.set_ylabel('v0')
ax.set_zlabel('Wurfweite')
ax.set_title('Wurfweite eines Körpers in Abhängigkeit von alpha und v0')
plt.show()

#zweidimensional mit den Höhenlinien
fig = plt.figure()
ax = fig.add_subplot(111)
X, Y = np.meshgrid(alpha, v0)
Z = W(Y, X)
contour = ax.contourf(X, Y, Z, cmap='viridis')
ax.set_xlabel('alpha')
ax.set_ylabel('v0')
ax.set_title('Wurfweite eines Körpers in Abhängigkeit von alpha und v0')
fig.colorbar(contour)
plt.show()


#b) Zustandsgleichnung pV = RT für ein Mol eines idealen Gases beschreibt den Zusammenhang zwischen den Grössen p (Druck in N/m^2), V (Volumen in m^3), T (Temperatur in K) und R (universelle Gaskonstante in J/(mol*K)):
R = 8.314

def pV(p, V, T):
    return R*T

#Daraus ergeben sich folgende Abhängigkeiten. Stellen sie jede der drei Funktionen dar innerhalb der angegebenen Definitionsbereiche für p, V, T
# p = p(V, T) = RT/V für V \in [0, 0.2] und T \in [0, 1e4]

V = np.linspace(0, 0.2, 100)
T = np.linspace(0, 1e4, 100)
X, Y = np.meshgrid(V, T)
Z = pV(R, X, Y)

#dreidimensional mit plot_wireframe()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('Volumen')
ax.set_ylabel('Temperatur')
ax.set_zlabel('Druck')
ax.set_title('Druck in Abhängigkeit von Volumen und Temperatur')
plt.show()

#dreidimensional mit plot_surface() und passender Colormap
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Volumen')
ax.set_ylabel('Temperatur')
ax.set_zlabel('Druck')
ax.set_title('Druck in Abhängigkeit von Volumen und Temperatur')
plt.show()

#zweidimensional mit den Höhenlinien
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(X, Y, Z, cmap='viridis')
ax.set_xlabel('Volumen')
ax.set_ylabel('Temperatur')
ax.set_title('Druck in Abhängigkeit von Volumen und Temperatur')
fig.colorbar(contour)
plt.show()

# V = V(p, T) = RT/p für p \in [1e4, 1e5] und T \in [0, 1e4]
p = np.linspace(1e4, 1e5, 100)
T = np.linspace(0, 1e4, 100)
X, Y = np.meshgrid(p, T)
Z = pV(X, R, Y)

#dreidimensional mit plot_wireframe()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('Druck')
ax.set_ylabel('Temperatur')
ax.set_zlabel('Volumen')
ax.set_title('Volumen in Abhängigkeit von Druck und Temperatur')
plt.show()

#dreidimensional mit plot_surface() und passender Colormap
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Druck')
ax.set_ylabel('Temperatur')
ax.set_zlabel('Volumen')
ax.set_title('Volumen in Abhängigkeit von Druck und Temperatur')
plt.show()

#zweidimensional mit den Höhenlinien
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(X, Y, Z, cmap='viridis')
ax.set_xlabel('Druck')
ax.set_ylabel('Temperatur')
ax.set_title('Volumen in Abhängigkeit von Druck und Temperatur')
fig.colorbar(contour)
plt.show()


# T = T(p, V) = pV/R für p \in [1e4, 1e6] und V \in [0, 10]
p = np.linspace(1e4, 1e6, 100)
V = np.linspace(0, 10, 100)
X, Y = np.meshgrid(p, V)
Z = pV(X, Y, R)

#dreidimensional mit plot_wireframe()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('Druck')
ax.set_ylabel('Volumen')
ax.set_zlabel('Temperatur')
ax.set_title('Temperatur in Abhängigkeit von Druck und Volumen')
plt.show()

#dreidimensional mit plot_surface() und passender Colormap
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Druck')
ax.set_ylabel('Volumen')
ax.set_zlabel('Temperatur')
ax.set_title('Temperatur in Abhängigkeit von Druck und Volumen')
plt.show()

#zweidimensional mit den Höhenlinien
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.contourf(X, Y, Z, cmap='viridis')
ax.set_xlabel('Druck')
ax.set_ylabel('Volumen')
ax.set_title('Temperatur in Abhängigkeit von Druck und Volumen')
fig.colorbar(contour)
plt.show()
