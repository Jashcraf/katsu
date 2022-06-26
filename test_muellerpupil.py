import numpy as np
import matplotlib.pyplot as plt
import mueller as mul
from polarimetry import *
import poppy
import astropy.units as u

# Load in some PRT Data, let's start with a POPPY-made pupil
primary = poppy.CircularAperture(radius=4)
secondary = poppy.AsymmetricSecondaryObscuration(secondary_radius=1*u.m,
                                             support_angle=(10,80,190,260),
                                             support_width=[0.1, 0.1, 0.1, 0.1],
                                             support_offset_x=[0, 0, 0, 0],
                                             name='Complex secondary')

aperture = poppy.CompoundAnalyticOptic(opticslist=[primary,secondary])

# plt.figure()
# aperture.display()
# plt.show()

wf = poppy.FresnelWavefront(wavelength=2.2e-6*u.m,beam_radius=4*u.m,npix=64,oversample=1)
array = aperture.get_transmission(wf)

# Some radiometric calculations
# Let's just call it detector counts!

array *= 6000
array = np.random.poisson(array)
print(array.shape[0])
array_raveled = np.ravel(array)

plt.figure(figsize=[4,4])
plt.title('Flux on Detector [counts]')
plt.imshow(array,extent=[-4,4,-4,4])
plt.xlabel('[m]')
plt.ylabel('[m')
plt.colorbar(label='counts')
plt.show()

x = np.linspace(-4,4,array.shape[0])
y = np.linspace(-4,4,array.shape[0])
x,y = np.meshgrid(x,y)

Min = mul.LinearRetarder(np.random.random()*np.pi,np.pi/2) @ mul.LinearPolarizer(np.random.random()*np.pi) @ mul.LinearRetarder(np.random.random(),np.pi/2)
Mpupil = np.zeros([4,4,array_raveled.shape[0]])

# Now do Mueller Polarimetry on Each Pixel
for i in range(len(array_raveled)):

        Mpupil[:,:,i] = FullMuellerPolarimeterMeasurement(Min,80,array_raveled[i])

mul.PlotMuellerArray(Mpupil,64,Min=Min)


