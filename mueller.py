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

