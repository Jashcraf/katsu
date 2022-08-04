import numpy as np
from polarimetry import *
import mueller as mul
import matplotlib.pyplot as plt

Min = mul.LinearRetarder(np.random.random()*np.pi,np.pi/2) @ mul.LinearPolarizer(np.random.random()*np.pi) @ mul.LinearRetarder(np.random.random(),np.pi/2)
print('Mueller Matrix to Measure')
print(Min)

condition_number = []

m00 = []
m01 = []
m02 = []
m03 = []

m10 = []
m11 = []
m12 = []
m13 = []

m20 = []
m21 = []
m22 = []
m23 = []

m30 = []
m31 = []
m32 = []
m33 = []



nmeas = np.arange(27,160,1)

for i,val in enumerate(nmeas):

    Mout,cond = FullMuellerPolarimeterMeasurement(Min,val,power=1,return_condition_number=True)
    Mout = np.reshape(Mout,[4,4])

    condition_number.append(cond)

    pDiff = (Mout-Min)/Min

    m00.append(pDiff[0,0])
    m01.append(pDiff[0,1])
    m02.append(pDiff[0,2])
    m03.append(pDiff[0,3])

    m10.append(pDiff[1,0])
    m11.append(pDiff[1,1])
    m12.append(pDiff[1,2])
    m13.append(pDiff[1,3])

    m20.append(pDiff[2,0])
    m21.append(pDiff[2,1])
    m22.append(pDiff[2,2])
    m23.append(pDiff[2,3])

    m30.append(pDiff[3,0])
    m31.append(pDiff[3,1])
    m32.append(pDiff[3,2])
    m33.append(pDiff[3,3])


plt.figure(figsize=[10,5])

# Condition Number Plot
plt.subplot(121)
plt.plot(nmeas,condition_number)
# plt.yscale('log')
plt.xlabel('# of Measurements')
plt.ylabel('Condition #')
plt.title('Condition # v.s. Measurements')

# % Difference Plot
plt.subplot(122)
plt.plot(nmeas,m00,label='M00')
plt.plot(nmeas,m01,label='M01')
plt.plot(nmeas,m02,label='M02')
plt.plot(nmeas,m03,label='M03')

plt.plot(nmeas,m10,label='M10')
plt.plot(nmeas,m11,label='M11')
plt.plot(nmeas,m12,label='M12')
plt.plot(nmeas,m13,label='M13')

plt.plot(nmeas,m20,label='M20')
plt.plot(nmeas,m21,label='M21')
plt.plot(nmeas,m22,label='M22')
plt.plot(nmeas,m23,label='M23')

plt.plot(nmeas,m30,label='M30')
plt.plot(nmeas,m31,label='M31')
plt.plot(nmeas,m32,label='M32')
plt.plot(nmeas,m33,label='M33')

plt.ylabel('% Diff in Mueller Element')
plt.xlabel('# of Measurement')
plt.legend()
plt.show()
