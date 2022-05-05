import numpy as np
from polarimetry import DualTetrahedronPolarimeter
import polutils as pol
from polarimetry import *

mueller_test =  pol.LinearPolarizerM(0) @ pol.LinearRetarderM(np.pi/4,np.pi/4)

mueller_reconstructed = DualTetrahedronPolarimeter(mueller_test)
mueller_reshaped = np.zeros([4,4])
mueller_reshaped[0,:] = mueller_reconstructed[0:4]
mueller_reshaped[1,:] = mueller_reconstructed[4:8]
mueller_reshaped[2,:] = mueller_reconstructed[8:12]
mueller_reshaped[3,:] = mueller_reconstructed[12:16]

print('Mueller Matrix out')
print(mueller_reshaped)


