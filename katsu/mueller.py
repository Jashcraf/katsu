import numpy as np

# The Parent Functions
def linear_polarizer(a):
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

    ones = np.ones_like(a)
    zeros = np.zeros_like(a)

    M01 = np.cos(2*a)
    M02 = np.sin(2*a)

    M10 = np.cos(2*a)
    M11 = np.cos(2*a)**2
    M12 = np.cos(2*a)*np.sin(2*a)
    
    M20 = np.sin(2*a)
    M21 = np.cos(2*a)*np.sin(2*a)
    M22 = np.sin(2*a)**2

    M = 0.5*np.array([[ones,M01,M02,zeros],
                    [M10,M11,M12,zeros],
                    [M20,M21,M22,zeros],
                    [zeros,zeros,zeros,zeros]])
    
    if M.ndim > 2:
        for _ in range(M.ndim - 2):
            M = np.moveaxis(M,-1,0)

    return M 

def linear_retarder(a,r):
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

    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    r = np.full_like(a,r)

    M11 = np.cos(2*a)**2 + np.cos(r)*np.sin(2*a)**2 # checked
    M12 = (1-np.cos(r))*np.cos(2*a)*np.sin(2*a) # checked
    M13 = -np.sin(r)*np.sin(2*a) # checked

    M21 = M12 # checked but uncertain
    M22 = np.cos(r)*np.cos(2*a)**2 + np.sin(2*a)**2 # checked
    M23 = np.cos(2*a)*np.sin(r) # checked

    M31 = -M13 # checked
    M32 = -M23 # checked
    M33 = np.cos(r) # checked

    M = np.array([[ones,zeros,zeros,zeros],
                [zeros,M11,M12,M13],
                [zeros,M21,M22,M23],
                [zeros,M31,M32,M33]])

    if M.ndim > 2:
        for _ in range(M.ndim - 2):
            M = np.moveaxis(M,-1,0)

    return M

def linear_diattenuator(a,Tmin):

    A = 1 + Tmin
    B = 1 - Tmin
    C = 2*np.sqrt(Tmin)

    zeros = np.zeros_like(a)

    M01 = B*np.cos(2*a)
    M02 = B*np.sin(2*a)

    M10 = M01
    M11 = A*np.cos(2*a)**2 + C*np.sin(2*a)**2
    M12 = (A-C) * np.cos(2*a) * np.sin(2*a)
    
    M20 = M02 
    M21 = M12
    M22 = C*np.cos(2*a)**2 + A*np.sin(2*a)**2

    M = np.array([[A,M01,M02,zeros],
                  [M10,M11,M12,zeros],
                  [M20,M21,M22,zeros],
                  [zeros,zeros,zeros,C]])
    
    if M.ndim > 2:
        for _ in range(M.ndim-2):
            M = np.moveaxis(M,-1,0)

    return M



