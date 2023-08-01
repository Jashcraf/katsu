import numpy as np
import mueller as mul
import polutils as pol

sys_test = mul.LinearRetarder(np.random.random()*np.pi,np.random.random()*2*np.pi) @ mul.LinearPolarizer(np.random.random()*np.pi)


nmeas = 10000

states = np.empty([4,nmeas])
measurements = np.empty([nmeas,4])

# populate num measurements
for i in range(nmeas):

    states[:,i] = np.array([1,np.random.random(),np.random.random(),0])
    measurements[i,:] = sys_test @ mul.StokesRandom()

print(measurements.shape)
print(states.shape)
print(sys_test/sys_test[0,0])
print(np.linalg.pinv(states).shape)
mulmeas = np.transpose(measurements) @ np.linalg.pinv(states)
print(mulmeas/mulmeas[0,0])