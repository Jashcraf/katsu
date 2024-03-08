import numpy as np

def _empty_mueller(shape):
    """Returns an empty array to populate with Mueller matrix elements.

    Parameters
    ----------
    shape : list
        shape to prepend to the mueller matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    numpy.ndarray
        The zero array of specified shape

    Notes
    -----
    The structure of this function was taken from prysm.x.polarization, which was written by Jaren Ashcraft
    """

    if shape is None:

        shape = (4, 4)

    else:

        shape = (*shape, 4, 4)

    return np.zeros(shape)

def linear_polarizer(a, shape=None):
    """returns a homogenous linear polarizer

    Parameters
    ----------
    a : float, or numpy.darray
        angle of the transmission axis w.r.t. horizontal in radians. If numpy array, must be the same shape as `shape`
    shape : list, optional
        shape to prepend to the mueller matrix array, see `_empty_mueller`. by default None

    Returns
    -------
    numpy.ndarray
        linear polarizer array
    """

    # returns zeros
    M = _empty_mueller(shape)

    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    cos2a = np.cos(2 * a)
    sin2a = np.sin(2 * a)

    M01 = np.cos(2*a)
    M02 = np.sin(2*a)

    M10 = np.cos(2*a)
    M11 = np.cos(2*a)**2
    M12 = np.cos(2*a)*np.sin(2*a)
    
    M20 = np.sin(2*a)
    M21 = np.cos(2*a)*np.sin(2*a)
    M22 = np.sin(2*a)**2

    # fist row
    M[..., 0, 0] = ones
    M[..., 0, 1] = M01
    M[..., 0, 2] = M02

    # second row
    M[..., 1, 0] = cos2a
    M[..., 1, 1] = cos2a**2
    M[..., 1, 2] = cos2a * sin2a

    # third row
    M[..., 2, 0] = sin2a
    M[..., 2, 1] = cos2a * sin2a
    M[..., 2, 2] = sin2a**2

    # Apply Malus' law directly to the forehead
    M /= 2  

    return M

def linear_retarder(a, r, shape=None):
    """returns a homogenous linear retarder

    Parameters
    ----------
    a : float, or numpy.ndarray
        angle of the fast axis w.r.t. horizontal in radians. If numpy array, must be the same shape as `shape`
    r : float, or numpy.ndarray
        retardance in radians. If numpy array, must be the same shape as `shape`
    shape : list, optional
        shape to prepend to the mueller matrix array, see `_empty_mueller`. by default None

    Returns
    -------
    numpy.ndarray
        linear retarder array
    """

    # returns zeros
    M = _empty_mueller(shape)

    # make sure everything is the right size
    if M.ndim > 2:
        a = np.broadcast_to(a, [*M.shape[:-2]])
        r = np.broadcast_to(r, [*M.shape[:-2]])

    # First row
    M[..., 0, 0] = 1.

    # second row
    M[..., 1, 1] = np.cos(2*a)**2 + np.cos(r)*np.sin(2*a)**2
    M[..., 1, 2] = (1-np.cos(r))*np.cos(2*a)*np.sin(2*a)
    M[..., 1, 3] = -np.sin(r)*np.sin(2*a)

    # third row
    M[..., 2, 1] = M[..., 1, 2] 
    M[..., 2, 2] = np.cos(r)*np.cos(2*a)**2 + np.sin(2*a)**2
    M[..., 2, 3] = np.cos(2*a)*np.sin(r)

    M[..., 3, 1] = -1 * M[..., 1, 3] # checked
    M[..., 3, 2] = -1 * M[..., 2, 3] # checked
    M[..., 3, 3] = np.cos(r) # checked

    return M

def linear_diattenuator(a, Tmin, Tmax=1, shape=None):
    """returns a homogenous linear diattenuator

    CLY 6.54

    Parameters
    ----------
    a : float, or numpy.ndarray
        angle of the transmission axis w.r.t. horizontal in radians. If numpy array, must be the same shape as `shape`
    Tmin : float, or numpy.ndarray
        Minimum transmission of the state orthogonal to maximum transmission. If numpy array, must be the same shape as `shape`
    shape : list, optional
        shape to prepend to the mueller matrix array, see `_empty_mueller`. by default None

    Returns
    -------
    numpy.ndarray
        linear diattenuator array
    """

    # returns zeros
    M = _empty_mueller(shape)

    # make sure everything is the right size
    if M.ndim > 2:
        a = np.broadcast_to(a, [*M.shape[:-2]])
        Tmin = np.broadcast_to(Tmin, [*M.shape[:-2]])

    A = Tmax + Tmin
    B = Tmax - Tmin
    C = 2 * np.sqrt(Tmax * Tmin)
    cos2a = np.cos(2 * a)
    sin2a = np.sin(2 * a)

    # first row
    M[..., 0, 0] = A
    M[..., 0, 1] = B*cos2a
    M[..., 0, 2] = B*sin2a

    # second row
    M[..., 1, 0] = M[..., 0, 1]
    M[..., 1, 1] = (A * cos2a**2) + (C * sin2a**2)
    M[..., 1, 2] = (A - C) * cos2a * sin2a 

    # third row
    M[..., 2, 0] = M[..., 0, 2]
    M[..., 2, 1] = M[..., 1, 2]
    M[..., 2, 2] = (C * cos2a**2) + (A * sin2a**2)

    # fourth row
    M[..., 3, 3] = C

    # Apply Malus' law directly to the forehead
    M /= 2  

    return M

# The depreciated parent functions from when katsu was Observatory-Polarimetry
def _linear_polarizer(a):
    """generates an ideal polarizer, depreciated

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

def _linear_retarder(a,r):
    """Generates an ideal retarder

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

def _linear_diattenuator(a,Tmin):
    """Generates an ideal diattenuator

    Parameters
    ----------
    a : float
        angle of the high-transmission axis w.r.t. horizontal in radians
    Tmin : float
        fractional transmission of the low-transmission axis

    Returns
    -------
    numpy.ndarray
        Mueller Matrix for Linear Diattenuator
    """

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
                  [zeros,zeros,zeros,C]]) / 2
    
    if M.ndim > 2:
        for _ in range(M.ndim-2):
            M = np.moveaxis(M,-1,0)

    return M



