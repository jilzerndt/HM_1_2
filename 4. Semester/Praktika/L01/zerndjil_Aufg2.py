import numpy as np
import matplotlib.pyplot as plt

'''
Plotte die Funktionen w(x, t) und v(x, t) drei-dimensional mit plot_wireframe() 
Annahme: c = 1
'''

#Funktion w(x, t):
def w(x, t):
    c = 1
    return np.sin(x + c*t)

#Funktion v(x, t):
def v(x, t):
    c = 1
    return np.sin(x + c*t) + np.cos(2*x + 2*c*t)


#plottin' time
#Definitionsbereich für x:  
x = np.linspace(0, 10, 100)
#Definitionsbereich für t:
t = np.linspace(0, 10, 100)

#dreidimensional mit plot_wireframe()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, t)
Z = w(X, Y)
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('w(x, t)')
ax.set_title('Funktion w(x, t)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, t)
Z = v(X, Y)
ax.plot_wireframe(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('v(x, t)')
ax.set_title('Funktion v(x, t)')
plt.show()
