import numpy as np
import mueller as mul



def TestLinearPolarizer():

    hpol = np.array([[1,1,0,0],
                     [1,1,0,0],
                     [0,0,0,0],
                     [0,0,0,0]])/2

    hpol_test = mul.LinearPolarizer(0)

    print(hpol == hpol_test)
    print(hpol)
    print(hpol_test)

    vpol = np.array([[1,-1,0,0],
                     [-1,1,0,0],
                     [0,0,0,0],
                     [0,0,0,0]])/2

    vpol_test = mul.LinearPolarizer(np.pi/2)

    print(vpol == vpol_test)
    print(vpol)
    print(vpol_test)

    ppol = np.array([[1,0,1,0],
                     [0,0,0,0],
                     [1,0,1,0],
                     [0,0,0,0]])/2

    ppol_test = mul.LinearPolarizer(np.pi/4)

    print(ppol == ppol_test)
    print(ppol)
    print(ppol_test)

    mpol = np.array([[1,0,-1,0],
                     [0,0,0,0],
                     [-1,0,1,0],
                     [0,0,0,0]])/2

    mpol_test = mul.LinearPolarizer(np.pi/4+np.pi/2)

    print(mpol == mpol_test)
    print(mpol)
    print(mpol_test)

def TestLinearRetarder():

    qret = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,-1,0]])

    qret_test = mul.LinearRetarder(0,np.pi/2)

    print(qret == qret_test)
    print(qret)
    print(qret_test)





    


TestLinearRetarder()