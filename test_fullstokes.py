from polarimetry import *
import mueller as mul

Sin = mul.StokesRandom()

Sout = FullStokesPolarimeterMeasurement(Sin,11)

print('Stokes In')
print(Sin)
print('Stokes Measured')
print(Sout)

Sin_array = GenerateStokesArray(Sin,51)
Sout_array = FullStokesPolarimeter(Sin_array,11)

PlotStokesArray(Sout_array)