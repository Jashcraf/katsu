import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('bmh')

okabe_colorblind8 = ['#E69F00','#56B4E9','#009E73',
                     '#F0E442','#0072B2','#D55E00',
                     '#CC79A7','#000000']
okabe_colorblind8.reverse()

plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=okabe_colorblind8)


NMODES = [8, 16, 32, 64]
RUNTIMES_JAX = [29.718157300027087, 33.16371909994632, 35.30553690018132, 38.715851000044495]
RUNTIMES_NUMPY = [106.86078090011142, 229.84310650010593, 654.9272669998463, 1885.387258599978]

plt.figure()
plt.plot(NMODES, RUNTIMES_NUMPY, marker="x", linestyle="None", markersize=10, label="Numpy")
plt.plot(NMODES, RUNTIMES_JAX, marker="o", linestyle="None", markersize=10, label="Jax CPU")
plt.ylabel("Runtime, seconds")
plt.xlabel("Number of Zernike modes")
plt.title("Spatial Calibration Runtimes")
plt.legend()
# plt.yscale("log")
plt.show()