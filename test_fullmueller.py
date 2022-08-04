from cgi import test
import numpy as np
from polarimetry import *
import polutils as pol
import mueller as mul

Min = mul.LinearRetarder(np.random.random()*np.pi,np.pi/2) @ mul.LinearPolarizer(np.random.random()*np.pi) @ mul.LinearRetarder(np.random.random(),np.pi/2)
# Min /= Min[0,0]
print('Normalized Mueller In')
print(Min)

# Calibration Air Measurement
Mout_cal = FullMuellerPolarimeterMeasurement(np.identity(4),20,power=1)
# print('Air Measurement Pinv')
# Malt_cal = np.reshape(Malt_cal,[4,4])
# print(np.reshape(Malt_cal,[4,4]))

print('Air Measurement')
Mout_cal = np.reshape(Mout_cal,[4,4])
# Mout_cal /= Mout_cal[0,0]
print(Mout_cal)

# Measure the SUT
Mout = FullMuellerPolarimeterMeasurement(Min,20,power=1)

Mout = np.reshape(Mout,[4,4])

# Malt = np.reshape(Malt,[4,4])

# Mout /= Mout[0,0]
# Malt /= Malt[0,0]

# print('Normalized Mueller Out pinv')
# print(Malt)

print('Normalized Mueller Out')
print(Mout)

print('% Difference')
print(100*(Min-Mout)/Min)

# print('% Difference pinv')
# print(100*(Min-Malt)/Min)

print('Condition Number for Min Measure = ',ConditionNumber(Mout))
print('Condition Number for air Measure = ',ConditionNumber(Mout_cal))

