from polarimetry import *
import mueller as mul

Sin = mul.StokesRandom()

Sout = FullStokesPolarimeterMeasurement(Sin,101)

print('Stokes In')
print(Sin)
print('Stokes Measured')
print(Sout)

Sin_array = GenerateStokesArray(Sin,51)
Sout_array = FullStokesPolarimeter(Sin_array,101)

PlotStokesArray(Sout_array,Sin=Sin_array)