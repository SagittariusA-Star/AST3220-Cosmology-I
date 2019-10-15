import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats
from task4 import AgeOfTheUniverse as AtU 

"""This script iterates over a grid of density parameters
   to compare the resulting luminosity distance to the provided data."""

np.seterr(invalid = "raise")            # Raising error instead of setting value to NaN

h         = 0.7                         # Dimensionless Hubble parameter
Gpc       = 2.99792458/h                # H_0/c to Gpc conversion factor
N         = int(1e3)                    # Length of arrays
Om0       = np.linspace(0, 2, N)        # Matter density parameters to iterate over
Ol0       = np.linspace(0, 2, N)        # Vacuum energy density parammeters to iterate over
ChiSquare = np.zeros(shape = (N, N))    # Empty array to store chi^2
ModelAge  = np.zeros_like(ChiSquare)    # Empty array to store t_0

Universe  = AtU(N = 1e2, h = h)
M         = Universe.M                  # Lenght of provided data
error     = Universe.error              # Provided data and errors
zdata     = Universe.z_data             # ---------||---------
dLdata    = Universe.dL_data            # ---------||---------
"""
for i in range(N):
    for j in range(N):
        try: 
            Universe.Initialize(Om0[i], Ol0[j])
            dL, z           = Universe.LumDist(zdata)
            dL              = dL*(Gpc)
            element         = (dL - dLdata)**2
            element        /= (error)**2
            ChiSquare[i, j] = np.sum(element)
            ModelAge[i, j]  = Universe.CurrentAge()
            
        except ValueError:
            print(f"Skip i = {i} and {j} due to ValueError,\
                 corresponding to Om0 = {Om0[i]} and Ol0 = {Ol0[j]}. Inf value asigned to chi^2.")
            ChiSquare[i, j] = float("inf")
            continue
        except FloatingPointError:
            print(f"Skip i = {i} and {j} due to FloatingPointError,\
                 corresponding to Om0 = {Om0[i]} and Ol0 = {Ol0[j]}. Inf value asigned to chi^2.")
            ChiSquare[i, j] = float("inf")
            continue
"""
#np.save("ChiSquare_h068.npy", ChiSquare) # Saving chi^2 matrix as an .npy file
#np.save("ModelAge_h068.npy", ModelAge)   # Saving t_0 matrix as an .npy file


"""Plot code (for the most) from this point on"""

ChiSquare = np.load("ChiSquare_h07.npy")
ModelAge = np.load("ModelAge_h07.npy")

pos = np.where(ChiSquare == np.min(ChiSquare))
Om0Best = float(Om0[pos[0]])
Ol0Best = float(Ol0[pos[1]])
Ok0Best = 1 - Om0Best - Ol0Best
print("-------------------------------")
print("Letting h = 0.7")
print("Best Om0: ", Om0Best)
print("Best Ol0: ", Ol0Best)
print("Best Ok0: ", Ok0Best)
print("Sum: ", Om0Best + Ol0Best + Ok0Best)


Universe.Initialize(Om0Best, Ol0Best)
dLBest = Universe.LumDist(zdata)[0]
dLBest *= Gpc
conf_95 = np.where(ChiSquare - np.min(ChiSquare) <= 6.17)
Chi_conf_95 = np.zeros_like(ChiSquare)
Chi_conf_95[conf_95] = ChiSquare[conf_95]

EdS = Universe.Initialize(1, 0)
dL_EdS = Universe.LumDist(zdata)[0]
dL_EdS *= Gpc

dS = Universe.Initialize(0, 1)
dL_dS = Universe.LumDist(zdata)[0]
dL_dS *= Gpc

Univ1 = Universe.Initialize(0.368, 0.784)
dL_Univ1 = Universe.LumDist(zdata)[0]
dL_Univ1 *= Gpc

Univ2 = Universe.Initialize(0.154, 0.497)
dL_Univ2 = Universe.LumDist(zdata)[0]
dL_Univ2 *= Gpc

Univ3 = Universe.Initialize(0.634, 0.202)
dL_Univ3 = Universe.LumDist(zdata)[0]
dL_Univ3 *= Gpc

plt.plot(zdata, dLBest, "k", label=rf"Best-fit: $\Omega_{{m0}} = {Om0Best:.2f}, \Omega_{{\Lambda 0}} = {Ol0Best:.2f}$", zorder = 1)
plt.plot(zdata, dL_Univ1, "orange", label=rf"$\Omega_{{m0}} = 0.368, \Omega_{{\Lambda 0}} = 0.784$", zorder = 2)
plt.plot(zdata, dL_Univ2, "r-.", label=rf"$\Omega_{{m0}} = 0.154, \Omega_{{\Lambda 0}} = 0.497$", zorder = 3)
plt.plot(zdata, dL_Univ3, "y--", label=rf"$\Omega_{{m0}} = 0.634, \Omega_{{\Lambda 0}} = 0.202$", zorder = 4)
plt.plot(zdata, dL_EdS, "blue", label="Einstein-de Sitter", zorder = 5)
plt.plot(zdata, dL_dS, "magenta", label="de Sitter", zorder  = 6)
plt.errorbar(zdata, dLdata, color = "g",  yerr = error, label = "Data", zorder = 7)
plt.xlabel(r"$z$", fontsize = 14)
plt.ylabel(r"$d_L$ [Gpc]", fontsize = 14)
plt.legend(loc = 0)
plt.grid()
plt.savefig("LumDistBest.jpg")

plt.figure()
X, Y = np.meshgrid(Ol0, Om0)
plt.contourf(X, Y, ChiSquare, 100, cmap = "gist_stern")
cbar = plt.colorbar()
cbar.set_label(r"$\chi^2$")
plt.plot(Ol0Best, Om0Best, "rx")
plt.contour(X, Y, ChiSquare - np.min(ChiSquare), levels=6.17, colors="blue")
plt.xlabel(r"$\Omega_{\Lambda 0}$", fontsize = 14)
plt.ylabel(r"$\Omega_{m 0}$", fontsize = 14)
plt.savefig("ContourChi.jpg")

plt.figure()
plt.title("Age of the Universe")
plt.contourf(X, Y, ModelAge, 100, cmap="gist_stern")
plt.colorbar()
plt.plot(Ol0Best, Om0Best, "rx")
plt.contour(X, Y, ChiSquare - np.min(ChiSquare), levels=6.17, colors="blue")
plt.xlabel(r"$\Omega_{\lambda 0}$")
plt.ylabel(r"$\Omega_{m 0}$")

t_0 = np.linspace(12, 18, 1000)

ChiSquare_h07 = ChiSquare 
ModelAge_h07  = ModelAge
conf_h07 = conf_95
Mode_h07 = ModelAge_h07[pos][0]/(3600*365*24*1e9)
PDF_h07 = stats.gaussian_kde(ModelAge_h07[conf_h07]/(3600*365*24*1e9))

ChiSquare_h068 = np.load("ChiSquare_h068.npy")
ModelAge_h068 = np.load("ModelAge_h068.npy")
conf_h068 = np.where(ChiSquare_h068 - np.min(ChiSquare_h068) <= 6.17)
pos_h068 = np.where(ChiSquare_h068 == np.min(ChiSquare_h068))
Mode_h068 = ModelAge_h068[pos_h068][0]/(3600*365*24*1e9)
PDF_h068 = stats.gaussian_kde(ModelAge_h068[conf_h068]/(3600*365*24*1e9))

ChiSquare_h074 = np.load("ChiSquare_h074.npy")
ModelAge_h074 = np.load("ModelAge_h074.npy")
conf_h074 = np.where(ChiSquare_h074 - np.min(ChiSquare_h074) <= 6.17)
pos_h074 = np.where(ChiSquare_h074 == np.min(ChiSquare_h074))
Mode_h074 = ModelAge_h074[pos_h074][0]/(3600*365*24*1e9)
PDF_h074 = stats.gaussian_kde(ModelAge_h074[conf_h074]/(3600*365*24*1e9))

n, bins = np.histogram(ModelAge_h07[conf_h07]/(3600*365*24*1e9), bins = 100)
print(f"h = 0.7  : t_0          = {Mode_h07:.5f} Gyr")
print(f"h = 0.7  : Average t_0  = {np.average(bins[1:], weights=n):.5f} Gyr")
print(f"h = 0.7  : rms t_0      = {np.average(bins[1:], weights=n**2):.5f} Gyr")
print("-------------------------------")


plt.figure()
plt.hist(ModelAge_h07[conf_h07]/(3600*365*24*1e9), bins=100, label = r"$h = 0.7$", alpha = 0.3)
plt.hist(ModelAge_h068[conf_h068]/(3600*365*24*1e9), bins=100, label = r"$h=0.68$", alpha = 0.3)
plt.hist(ModelAge_h074[conf_h074]/(3600*365*24*1e9), bins=100, label = r"$h=0.74$", alpha = 0.3)
plt.xlabel(r"$t_0$ [Gyr]", fontsize = 14)
plt.ylabel("# of models", fontsize = 14)
plt.legend(loc=0, fontsize = 14)
plt.savefig("HistPlot.jpg")
plt.show()
