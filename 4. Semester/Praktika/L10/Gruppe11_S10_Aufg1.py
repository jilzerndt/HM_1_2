import math

def romberg(f, a, b, m):
    R = [[0] * (i+1) for i in range(m+1)]

    for j in range(m+1):
        n = 2**j
        h = (b - a) / n
        summe = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            x = a + i * h
            summe += f(x)
        R[j][0] = h * summe

    for k in range(1, m+1):
        for j in range(k, m+1):
            R[j][k] = R[j][k-1] + (R[j][k-1] - R[j-1][k-1]) / (4**k - 1)

    return R[m][m]

# Teil a) Stillstandszeit
def integrand_t(v):
    m = 97000
    return m / (5 * v**2 + 570000)

t_E = romberg(integrand_t, 0, 100, 5)
print("Stillstandszeit t_E:", t_E, "s")

# Teil b) Bremsweg
def integrand_x(v):
    m = 97000
    return (m * v) / (5 * v**2 + 570000)

x_E = romberg(integrand_x, 0, 100, 5)
print("Bremsweg x_E:", x_E, "m")
