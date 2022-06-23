import numpy as np
from polarimetry import *
import polutils as pol
import mueller as mul

Min = mul.LinearRetarder(np.random.random()*2*np.pi,np.pi/2) @ mul.LinearPolarizer(np.random.random()*2*np.pi) @ mul.LinearRetarder(np.random.random()*2*np.pi,np.pi/2)
Min /= Min[0,0]
print('Mueller In')
print(Min)

Mout,Malt = FullMuellerPolarimeterMeasurement(Min,80)

Malt = np.reshape(Malt,[4,4])

Mout /= Mout[0,0]
Malt /= Malt[0,0]

# print('Mueller Out Fourier')
# print(Mout)

print('Mueller Out pinv')
print(Malt)

# print('Difference Fourier')
# print(Min-Mout)

print('% Difference pinv')
print(100*(Min-Malt)*Min)

# print('Mueller Out')
# print(Malt)