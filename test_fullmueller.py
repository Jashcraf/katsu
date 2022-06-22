import numpy as np
from polarimetry import *
import polutils as pol

Min = np.random.random(size=[4,4])

print('Mueller In')
print(Min)

Mout = FullMuellerPolarimeterMeasurement(Min,101)

print('Mueller Out')
print(Mout)