from termios import FF1
import numpy as np
import matplotlib.pyplot as plt

def Lens(efl):
    return np.array([[1,0],
                     [-1/efl,1]])

def Distance(d):
    return np.array([[1,d],
                     [0,1]])

dep = 100
dxp = 100
f1 = 200
f2 = -75
d1 = 165.2

system =  Lens(f2) @ Distance(d1) @ Lens(f1) @ Distance(dep)

# A = 1 - d1/f1 + dxp*(-1/f2 * (1 - d1/f1) - 1/f1)
# B = dep + d1*(1 - dep/f1) + dxp*(-1/f2 * (dep + d1*(1-dep/f1)) + 1 - dep/f1)
# C = -1/f2 * (1-d1/f1) - 1/f1 
# D = -1/f2*(dep + d1*(1-dep/f1)) + 1 - dep/f1 

A = 1-d1/f1
B = dep + d1*(1-dep/f1)
C = -1/f2 * (1-d1/f1) - 1/f1
D = -1/f2*(dep + d1*(1-dep/f1)) + 1 - dep/f1

sys = np.array([[A,B],[C,D]]) @ np.array([[0],[.008]])

print(system)
print(sys)