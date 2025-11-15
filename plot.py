import numpy as np

Xk = np.array([[0.4321513567006913, 0.5416490884193732], [0.026199554879935444, 0.0]])

a = np.array([0.9738004451200646, 0.026199554879935444])

b = np.array([0.4583509453392474, 0.5416490546607525])

C = np.array([[0.4985070123025904, 0.22479664553084766], [0.19806286475962398, 0.7605307121989587]])

eta = 1e-3


Xk = np.array([[0.1587474148862999, 0.2787598513544877], [0.20409779280661933, 0.358394940952593]])
a = np.array([0.4375072662407877, 0.5624927337592124])
b = np.array([0.3628452076929192, 0.6371547923070807])
C = np.array([[0.8929469543476547, 0.8962930889334381], [0.12558531046383625, 0.20724287813818676]])
eta = 1e-3


Xk = np.array([[0.5764054060880065, 0.06220916833482073], [0.0, 0.3613854255771847]])
a = np.array([0.6386145744228153, 0.3613854255771847])
b = np.array([0.576405425852031, 0.4235945741479689])
C = np.array([[0.0, 1.0], [1.0, 0.0]])
eta = 1e-3

# a = np.array([0.5, 0.5])
# b = np.array([0.5, 0.5])
# C = np.array([[0.0, 1.0], [1.0, 0.0]])
# Xk = np.array([[0.3, 0.2], [0.2, 0.3]])
# eta = 0.1
# Let's plot S^k(gamma) for m=n=2 with a concrete, editable setup.
# You can change a, b, C, Xk, and eta below.

import matplotlib.pyplot as plt

assert a.shape == (2,) and b.shape == (2,)
assert C.shape == (2,2) and Xk.shape == (2,2)
assert np.all(a >= 0) and np.isclose(a.sum(), 1), "a must be a prob. vector"
assert np.all(b >= 0) and np.isclose(b.sum(), 1), "b must be a prob. vector"
assert np.all(Xk >= 0), "X^k must be nonnegative"
assert eta > 0

def S_k(gamma):
    """
    S^k(gamma) = sum_j gamma_j b_j - eta * sum_i a_i * log( sum_j X_ij^k * exp((gamma_j - C_ij)/eta) )
    gamma: shape (2,)
    """
    gamma = np.asarray(gamma)
    term1 = np.dot(gamma, b)
    # stable log-sum-exp per i
    # For each i, compute logsumexp over j of log(X_ij^k) + (gamma_j - C_ij)/eta
    logs = np.log(Xk) + (gamma[None, :] - C) / eta  # shape (2,2)
    # log-sum-exp rowwise
    m_i = np.max(logs, axis=1, keepdims=True)                # shape (2,1)
    lse = m_i.squeeze() + np.log(np.sum(np.exp(logs - m_i), axis=1))  # shape (2,)
    term2 = eta * np.dot(a, lse)
    return -term1 + term2

    fixed_term = Xk * np.exp(-C / eta)  # (m,n)
    max_gamma = np.max(gamma)
    e = np.exp((gamma - max_gamma) / eta)  # (n,)
    numerator = fixed_term * e  # (m,n)
    row_sum = numerator.sum(axis=1)  # (m,)

    return -np.dot(gamma, b) + eta * np.sum(a * np.log(row_sum)) + max_gamma

r1 = S_k(np.array([-0.5, 0.5]))
r2 = S_k(np.array([-0.5-1e-8, 0.5]))
r3 = S_k(np.array([-0.5, 0.5+1e-8]))

print(r1)
print(r2)
print(r3)

print("derivative at (-0.5, 0.5) is ", (r2-r1)/1e-8, (r3-r1)/1e-8)
# Build grid over gamma = (g1, g2)
gmin, gmax, num = -0.75, 0.75, 1000
g1 = np.linspace(gmin, gmax, num)
g2 = np.linspace(gmin, gmax, num)
G1, G2 = np.meshgrid(g1, g2)
Z = np.zeros_like(G1)

for i in range(num):
    for j in range(num):
        Z[i, j] = S_k(np.array([G1[i, j], G2[i, j]]))

# Plot: contour and 3D surface (separate figures)
plt.figure(figsize=(6,5))
cs = plt.contour(G1, G2, Z, levels=20)
plt.clabel(cs, inline=True, fontsize=8)
plt.plot([-0.5, -0.5], [-0.75, 0.75], color='red')
plt.plot([-0.75, 0.75], [0.5, 0.5], color='red')
plt.xlabel(r'$\gamma_1$')
plt.ylabel(r'$\gamma_2$')
plt.title(r'$S^k(\gamma)$ contour (m=n=2)')
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(G1, G2, Z, linewidth=0, antialiased=True)
ax.set_xlabel(r'$\gamma_1$')
ax.set_ylabel(r'$\gamma_2$')
ax.set_zlabel(r'$S^k(\gamma)$')
ax.set_title(r'3D surface of $S^k(\gamma)$ (m=n=2)')
plt.tight_layout()
plt.show()