import numpy as np

# The Parent Functions
def LinearPolarizer(a):
    """Quinn Jarecki's Linear Polarizer, generates an ideal polarizer

    CLY Eq 6.37
    checked!

    Parameters
    ----------
    a : float
       angle of transmission axis w.r.t. horizontal in radians

    Returns
    -------
    numpy.ndarray
        Mueller Matrix for the linear polarizer
    """

    M00 = 1
    M01 = np.cos(2*a)
    M02 = np.sin(2*a)

    M10 = np.cos(2*a)
    M11 = np.cos(2*a)**2
    M12 = np.cos(2*a)*np.sin(2*a)
    
    M20 = np.sin(2*a)
    M21 = np.cos(2*a)*np.sin(2*a)
    M22 = np.sin(2*a)**2

    return 0.5*np.array([[1,M01,M02,0],
                         [M10,M11,M12,0],
                         [M20,M21,M22,0],
                         [0,0,0,0]])

def LinearRetarder(a,r):
    """Quinn Jarecki's Linear Retarder, generates an ideal retarder

    Parameters
    ----------
    a : float
        angle of fast axis w.r.t. horizontal in radians
    r : float
        retardance in radians

    Returns
    -------
    numpy.ndarray
        Mueller Matrix for Linear Retarder
    """

    M11 = np.cos(2*a)**2 + np.cos(r)*np.sin(2*a)**2 # checked
    M12 = (1-np.cos(r))*np.cos(2*a)*np.sin(2*a) # checked
    M13 = -np.sin(r)*np.sin(2*a) # checked

    M21 = M12 # checked but uncertain
    M22 = np.cos(r)*np.cos(2*a)**2 + np.sin(2*a)**2 # checked
    M23 = np.cos(2*a)*np.sin(r) # checked

    M31 = -M13 # checked
    M32 = -M23 # checked
    M33 = np.cos(r) # checked

    return np.array([[1,0,0,0],
                     [0,M11,M12,M13],
                     [0,M21,M22,M23],
                     [0,M31,M32,M33]])

def CircularPolarizer(a):
    """Circular Polarizer

    Parameters
    ----------
    a : float
        angle of transmission axis w.r.t. horizontal in radians

    Returns
    -------
    numpy.ndarray
        Mueller Matrix for a Circular Polarizer
    """

    LP = LinearPolarizer(a)
    QP = LinearRetarder(a+np.pi/2,np.pi/2)

    return QP @ LP

def StokesFromPoincareAngle(lat,lon):
    S0 = 1
    S1 = np.cos(lat)*np.cos(lon)
    S2 = np.cos(lat)*np.sin(lon)
    S3 = np.sin(lat)
    return np.array([S0,S1,S2,S3])

def StokesH():
    lat = 0
    lon = 0
    return StokesFromPoincareAngle(lat,lon)

def StokesV():
    lat = 0
    lon = np.pi

def StokesRandom():
    S0 = 1
    S1 = -1 + 2*np.random.rand()
    S2 = -1 + 2*np.random.rand()
    S3 = -1 + 2*np.random.rand()
    return np.array([S0,S1,S2,S3])

def PlotMuellerArray(M,npix,vmin=-1,vmax=1,Min=None):

    import matplotlib.pyplot as plt

    plt.figure(figsize=[8,8])

    M00 = M[0,0,:]
    for i in range(len(M00)):
        M[:,:,i] /= M00[i]

        if Min is not None:
            Min /= Min[0,0]
            M[:,:,i] = 100*(M[:,:,i] - Min)/Min

    if Min is not None:
        print('Mueller In')
        print(Min)

    M00 = np.reshape(M[0,0,:],[npix,npix])
    M01 = np.reshape(M[0,1,:],[npix,npix])
    M02 = np.reshape(M[0,2,:],[npix,npix])
    M03 = np.reshape(M[0,3,:],[npix,npix])

    M10 = np.reshape(M[1,0,:],[npix,npix])
    M11 = np.reshape(M[1,1,:],[npix,npix])
    M12 = np.reshape(M[1,2,:],[npix,npix])
    M13 = np.reshape(M[1,3,:],[npix,npix])

    M20 = np.reshape(M[2,0,:],[npix,npix])
    M21 = np.reshape(M[2,1,:],[npix,npix])
    M22 = np.reshape(M[2,2,:],[npix,npix])
    M23 = np.reshape(M[2,3,:],[npix,npix])

    M30 = np.reshape(M[3,0,:],[npix,npix])
    M31 = np.reshape(M[3,1,:],[npix,npix])
    M32 = np.reshape(M[3,2,:],[npix,npix])
    M33 = np.reshape(M[3,3,:],[npix,npix])

    # M00
    plt.subplot(4,4,1)
    plt.imshow(M00,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.ylabel('[m]')

    # M01
    plt.subplot(4,4,2)
    plt.imshow(M01,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M02
    plt.subplot(4,4,3)
    plt.imshow(M02,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M03
    plt.subplot(4,4,4)
    plt.imshow(M03,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M10
    plt.subplot(4,4,5)
    plt.imshow(M10,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.ylabel('[m]')

    # M11
    plt.subplot(4,4,6)
    plt.imshow(M11,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M12
    plt.subplot(4,4,7)
    plt.imshow(M12,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M13
    plt.subplot(4,4,8)
    plt.imshow(M13,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M20
    plt.subplot(4,4,9)
    plt.imshow(M20,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.ylabel('[m]')

    # M21
    plt.subplot(4,4,10)
    plt.imshow(M21,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M22
    plt.subplot(4,4,11)
    plt.imshow(M22,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M23
    plt.subplot(4,4,12)
    plt.imshow(M23,vmin=vmin,vmax=vmax)
    plt.colorbar()

    # M30
    plt.subplot(4,4,13)
    plt.imshow(M30,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.ylabel('[m]')
    plt.xlabel('[m]')

    # M31
    plt.subplot(4,4,14)
    plt.imshow(M31,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.xlabel('[m]')

    # M32
    plt.subplot(4,4,15)
    plt.imshow(M32,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.xlabel('[m]')

    # M33
    plt.subplot(4,4,16)
    plt.imshow(M33,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.xlabel('[m]')

    if Min is not None:
        plt.suptitle('% Difference to input Mueller Matrix')
    plt.show()





