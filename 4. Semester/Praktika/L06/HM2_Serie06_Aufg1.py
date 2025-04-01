
import numpy as np
import matplotlib.pyplot as plt

# a) fit f(T) = aT^2 + bT + c to data using least squares

T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
rho = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4])

# matrix A
A = np.column_stack((T**2, T, np.ones(len(T))))

# direct normal equations
ATA = A.T @ A
ATy = A.T @ rho
lambda_direct = np.linalg.solve(ATA, ATy)

print('a) Coefficients using direct normal equations:')
print(f'a = {lambda_direct[0]}')
print(f'b = {lambda_direct[1]}')
print(f'c = {lambda_direct[2]}')

# QR Zerlegung
Q, R = np.linalg.qr(A)
lambda_qr = np.linalg.solve(R, Q.T @ rho)

print('\nCoefficients using QR:')
print(f'a = {lambda_qr[0]}')
print(f'b = {lambda_qr[1]}')
print(f'c = {lambda_qr[2]}')

# plotting time
T_fine = np.linspace(0, 100, 1000)
f_direct = lambda_direct[0]*T_fine**2 + lambda_direct[1]*T_fine + lambda_direct[2]
f_qr = lambda_qr[0]*T_fine**2 + lambda_qr[1]*T_fine + lambda_qr[2]

plt.figure(figsize=(10, 6))
plt.scatter(T, rho, color='red', marker='o', label='Data points')
plt.plot(T_fine, f_direct, 'b-', linewidth=2, label='Direct method')
plt.grid(True)
plt.xlabel('Temperature (°C)')
plt.ylabel('Density (g/l)')
plt.title('Water Density vs. Temperature')
plt.legend()

# b) comp condition numbers
cond_ATA = np.linalg.cond(ATA)
cond_R = np.linalg.cond(R)

print('\nb) Condition numbers:')
print(f'Condition nr of A^T A: {cond_ATA}')
print(f'Condition nr of R: {cond_R}')
print(f'The condition nr of A^T A is {cond_ATA/cond_R} times larger than that of R')
print('-> QR decomposition leads to better-conditioned problem')

# c) numpy.polyfit 
p = np.polyfit(T, rho, 2)

print('\nc) Coefficients using numpy.polyfit():')
print(f'a = {p[0]}')
print(f'b = {p[1]}')
print(f'c = {p[2]}')

# polyfit solution
f_polyfit = p[0]*T_fine**2 + p[1]*T_fine + p[2]
plt.plot(T_fine, f_polyfit, 'g--', linewidth=2, label='Polyfit')
plt.legend()
plt.savefig('water_density_fit.png', dpi=300, bbox_inches='tight')
plt.show()

# d) eroors
error_direct = np.sum((rho - (lambda_direct[0]*T**2 + lambda_direct[1]*T + lambda_direct[2]))**2)
error_qr = np.sum((rho - (lambda_qr[0]*T**2 + lambda_qr[1]*T + lambda_qr[2]))**2)
error_polyfit = np.sum((rho - (p[0]*T**2 + p[1]*T + p[2]))**2)

print('\nd) Error functionals:')
print(f'Error from direct method: {error_direct}')
print(f'Error from QR method: {error_qr}')
print(f'Erro from polyfit: {error_polyfit}')

# differences explaintation
if abs(error_direct - error_polyfit) < 1e-10 and abs(error_qr - error_polyfit) < 1e-10:
    print('All three methods yield essentially the same error (as expected for a well-conditioned problem)')
else:
    print('some small differencies in the errors, which may be due to numerical precision')

#calculate R^2 to assess goodness of fit
rho_mean = np.mean(rho)
SS_tot = np.sum((rho - rho_mean)**2)
SS_res = error_direct  
R_squared = 1 - SS_res/SS_tot

print(f'\nR-squared value: {R_squared}')
print(f'This indicates that ~ {R_squared*100}% of the variation in density')
print('is explained by the quadratic relationhip with temp')