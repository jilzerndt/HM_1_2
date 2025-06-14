def trapez(f, a, b, n):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i*h)
    return h * s

def integrate_cumulative(f, a, b, steps):
    result = []
    for i in range(1, steps+1):
        res = trapez(f, a, b, i)
        result.append(res)
    return result

# Parameter Ariane 4
v_rel = 2600
mA = 300000
mE = 80000
tE_rakete = 190
g = 9.81
mu = (mA - mE) / tE_rakete

# Beschleunigung als Funktion von t
def a(t):
    return v_rel * mu / (mA - mu * t) - g

# Geschwindigkeit v(t) = ∫ a(t) dt
def v(t):
    return trapez(a, 0, t, 100)

# Höhe h(t) = ∫ v(t) dt
def h(t):
    return trapez(v, 0, t, 100)

# Plot vorbereiten
import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(0, tE_rakete, 100)
acc_values = [a(t) for t in times]
vel_values = [trapez(a, 0, t, 100) for t in times]
height_values = [trapez(lambda tau: trapez(a, 0, tau, 100), 0, t, 100) for t in times]

# Plot Beschleunigung
plt.figure()
plt.plot(times, acc_values)
plt.title("Beschleunigung a(t)")
plt.xlabel("t [s]")
plt.ylabel("a(t) [m/s²]")
plt.grid()
plt.show()

# Plot Geschwindigkeit
plt.figure()
plt.plot(times, vel_values)
plt.title("Geschwindigkeit v(t)")
plt.xlabel("t [s]")
plt.ylabel("v(t) [m/s]")
plt.grid()
plt.show()

# Plot Höhe
plt.figure()
plt.plot(times, height_values)
plt.title("Höhe h(t)")
plt.xlabel("t [s]")
plt.ylabel("h(t) [m]")
plt.grid()
plt.show()

# Endwerte
final_v = vel_values[-1]
final_h = height_values[-1]
final_a = acc_values[-1]
print("Endgeschwindigkeit:", final_v, "m/s")
print("Endhöhe:", final_h, "m")
print("Endbeschleunigung:", final_a, "m/s² (~", final_a/g, "g)")
