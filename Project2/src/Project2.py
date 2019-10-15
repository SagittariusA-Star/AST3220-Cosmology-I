import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as ig

kB      = 1.380648e-23      # [J/K] Boltzmann constant
mc2     = 0.511*1.6022e-13  # [J] Electron rest mass energy
T0      = 1e11              # [K] Initial temperature
hbarc   = 3.161526e-26      # [Jm] Planck constant times light speed
c       = 299792458         # [m/s] Speed of light
G       = 6.674e-11         # [m^3kg^-1s^-2] Gravitational constant

def f(y, x):
    """Function returns integrand of S(x)"""
    I  = y**2
    I *= (np.sqrt(y**2 + x**2) + y**2/(3*np.sqrt(y**2 + x**2)))
    I /= (np.exp(np.sqrt(y**2 + x**2)) + 1)
    return I

def feps(y, x):
    """Function returns integrand of eps(x)"""
    I2  = y**2 * np.sqrt(y**2 + x**2)
    I2 /= (np.exp(np.sqrt(y**2 + x**2)) + 1)
    return I2

def S(x):
    """Function S(x) used to calculate the entropy
       density s(T). S(x) corresponds roughly to
       the effective relativistic degrees of the system
       due to photons, electrons and positrons."""
    integ = np.ones_like(x) * 45 / (2 * np.pi**4)
    for i in range(len(x)):
        integ[i]      = ig.quad(f, 0, 1e2, args = (x[i],))[0]
        integ[i]     += 1
    return integ

def DS(x):
    """Function returns dS(x)/dx."""
    return np.gradient(S(x))

def eps(x):
    """Function returns eps(x) used to calculate the
       energy density due to photons, neutrinos, electrons
       and positrons, and it's used to calculate the time
       after the Big Bang corresponding to a certain
       temperature (x = m_ec^2/kBT)."""
    Ieps  = np.ones_like(x)
    Ieps += 21 / 8 * (4/11)**(4/3) * S(x)**(4/3)
    for i in range(len(x)):
        Ieps[i] += 30 / np.pi**4 * ig.quad(feps, 0, np.inf, args=(x[i],))[0]
    return Ieps


def ft(x):
    """Function returns the integrand for
       the time function t."""
    I2 = x * (3 - DS(x) / S(x)) / np.sqrt(eps(x))
    return I2

def t(T):
    """Function returns the time passed since the
       Big Bang corresponding to a given
       photon temperature T."""
    x       = mc2 / (kB*T)  # Upper integration limit
    x0      = mc2 / (kB*T0) # Lower integration limit
    xarr    = np.linspace(x0, x, int(1e3)) # x-axis to integrate over
    time    = np.sqrt(15 * hbarc**3 * c**2 / (24 * np.pi**3 * G * mc2**4)) # Integral coefficient
    time    *= ig.trapz(ft(xarr), xarr)    # Integral
    return time

def Tnu(T):
    """Function returns the neutrino temperature T_nu
       for given photon temperature T."""
    x = mc2 / (kB*T)
    return (4/11)**(1/3)*T*(S(x))**(1/3)

"""Plot code start"""
x = np.linspace(0, 20, int(1e3))
y = S(x)

plt.figure()
plt.tight_layout(pad = 3.5)
plt.plot(x, y, label=r"$S(x = \frac{m_ec^2}{k_BT})$")
plt.xlabel(r"$x = m_ec^2/k_BT$", fontsize=12)
plt.ylabel(r"$S(x)$", fontsize = 12)
plt.grid()
plt.legend(loc = 0, fontsize = 12)
plt.savefig("plotoppg7.jpg")
plt.show()
"""Plot code end"""

"""Filling out table in Ex. 8. and printing it out"""

Tlist = (np.array([1e11, 6e10, 2e10, 1e10, 6e9,
                   3e9, 2e9, 1e9, 3e8, 1e8, 1e7, 1e6]))
Tnu  = Tnu(Tlist)
time = np.zeros_like(Tlist)

for i in range(len(Tlist)):
    time[i] = t(Tlist[i])
    print(f"T = {Tlist[i]:.4g}K T_nu/T = {Tnu[i]/Tlist[i]:.4g} t = {time[i]:.4g}s ")
