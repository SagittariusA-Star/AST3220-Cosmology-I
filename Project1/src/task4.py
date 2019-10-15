import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt


class AgeOfTheUniverse:
    km = 1e3        # [m]
    Mpc = 3.09e22   # [m]

    def __init__(self, N = 1e3, h=0.7, filename = "sndata.txt", skiprows = 5):
        self.h   = h                                  # Dimensionless Hubble constant
        self.H0  = 100 * self.h * self.km / self.Mpc  # [km s^-1 Mpc^-1] Hubble constant
        self.z_data, self.dL_data, self.error = self.ReadData(filename, skiprows=5) #Reading in data set
        self.M   = len(self.z_data)                   # Length of data set
        self.N   = int(N)                             # Length of arrays   
    
    def ReadData(self, filename, skiprows = 5):
        """Method reads supernova data into arrays.
           Returns redshift z, luminosity distance dL
           and the corresponding uncertainties called error."""
        Data  = np.loadtxt(filename, dtype=float, skiprows = skiprows)
        z     = Data[:, 0]
        dL    = Data[:, 1]
        error = Data[:, 2]
        return z, dL, error

    def Initialize(self, Om0, Ol0):
        """Method sets class attributes"""
        self.Om0 = Om0        # "Dust" density parameter
        self.Ol0 = Ol0        # Cosmological constant density parameter
        self.Ok0 = (1 - self.Om0 - self.Ol0)
        if np.any(self.FI(np.linspace(1e-15, 1, self.N)) <= 0):
            raise ValueError("Friedmann I returns zero r.h.s.!\
                    Please try other Om0, Ol0 and h.")
        if self.Ok0 < 0:
            self.k = 1

        elif self.Ok0 > 0:
            self.k = -1
        
        else:
            self.k = 0

    def FI(self, x):
        """Method returns r.h.s. of first Friedmann equation"""
        return (self.Om0 * x**-3
                + self.Ok0 * x**-2 + self.Ol0)

    def H(self, x):
        """Method returns Hubble parameter as a function
           of x = a/a_0"""
        return self.H0*np.sqrt(self.FI(x))

    def CurrentAge(self):
        """Method returns current age of the universe
           for given density parameters"""
        x         = np.linspace(0, 1, self.N)
        dx        = x[2] - x[1]
        Integrand = 1/(x[1:] * self.H(x[1:]))
        t0        = np.trapz(Integrand, dx=dx)
        return t0 #[s]

    def RedShift(self, x):
        """Method returns redshift z as function
           of x = a/a_0"""
        return 1/x - 1

    def TimeElapsed(self, x_f):
        """Method returns time elaped  since the Big Bang, 
           corresponding scalefactor x = a/a_0
           and corresponding redshift z."""
        x         = np.linspace(0, x_f, self.N)
        Integrand = 1/(x[1:] * self.H(x[1:]))
        t         = sci.cumtrapz(Integrand, x = x[1:], initial = 0)
        t        /= self.CurrentAge()
        z         = self.RedShift(x[1:])
        return t, x, z
    
    def LumDist(self, z_array):
        """Method returns the luminocity distance for
           given input array of redshifts. Each case of 
           spatial curvature is considered seperately 
           in the if-block."""
        z         = np.array([np.linspace(0, i, self.N) for i in z_array])  # Redshift
        HH0       = np.sqrt(self.Om0 * (1 + z)**3
                            + self.Ok0 * (1 + z)**2 + self.Ol0)             # H/H_0
        Integrand = np.trapz(1/HH0, x = z, axis = 1)
        dL        = (1 + z[:, -1])                                          # Luminosity distance
        
        if self.k == 1:
            Integrand *= np.sqrt(np.abs(self.Ok0))
            dL        *= np.sin(Integrand) / np.sqrt(np.abs(self.Ok0))
        elif self.k == -1:
            Integrand *= np.sqrt(np.abs(self.Ok0))
            dL        *= np.sinh(Integrand) / np.sqrt(np.abs(self.Ok0))
        else:
            dL        *= Integrand
        return dL, z[:, -1]
        
if __name__ == "__main__":
    """Sanity check by comparing numerical result to analytical
       result of Einstein-de Sitter universe (EdS)."""
    Om0 = 1         # Matter density parameter
    Ol0 = 0         # Vacuum energy density parameter
    N   = int(1e3)  # Length of arrays
    xf  = 1         # Final scale factor

    z_arr   = np.linspace(0, 5, N)  # Redshifts

    def Analytical_scale(t):
        """Analytical scale factor for EdS"""
        return np.power(t, 2/3)

    def dL_analytical(z):
        """Analytical luminosity distance for EdS"""
        return 2*(1 + z - np.sqrt(1 + z))

    """Generating arrays to plot from here on"""

    EdS          = AgeOfTheUniverse(N)
    EdS.Initialize(Om0, Ol0)
    t, x, z      = EdS.TimeElapsed(xf)
    dL, redshift = EdS.LumDist(z_arr)
    x_analy      = Analytical_scale(t)
    z_analy      = np.linspace(0, 5, int(N))
    dL_analy     = dL_analytical(z_analy)

    print("Current age: ", EdS.CurrentAge()/(3600*365*24*1e9), " Gyr")
    
    plt.subplots(1, 3, figsize = (14, 5))
    plt.tight_layout(pad = 2.5)

    plt.subplot(1, 3, 1)
    plt.title(r"$a/a_0$", fontsize = 14)
    plt.plot(t, x[1:], "b", label="Numerical")
    plt.plot(t, x_analy, "r--", label = "Analytical")
    plt.xlabel(r"$t/t_0$", fontsize = 14)
    plt.ylabel(r"$a/a_0$", fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.grid()
    plt.legend(loc=0, fontsize = 14)
    
    pos = np.logical_and(z >= 0, z <= 5)
    z = z[pos]
    t = t[pos]

    plt.subplot(1, 3, 2)
    plt.title(r"$t/t_0$", fontsize = 14)
    plt.plot(z, t, "b", label = "Numerical")
    plt.plot(z_analy, np.power(1 + z_analy, -3/2), "r--", label="Analytical")
    plt.xlabel(r"$z$", fontsize = 14)
    plt.ylabel(r"$t/t_0$", fontsize = 14)
    plt.grid()
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(loc=0, fontsize = 14)

    plt.subplot(1, 3, 3)
    plt.title(r"$d_L$", fontsize = 14)
    plt.plot(redshift, dL, "b", label="Numerical")
    plt.plot(z_analy, dL_analy, "r--", label="Analytical")
    plt.xlabel(r"$z$", fontsize = 14)
    plt.ylabel(r"$d_L [c/H_0]$", fontsize = 14)
    plt.grid()
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(loc=0, fontsize = 14)
    #plt.savefig("EdSPlot.jpg")
    plt.show()
    
