import numpy as np
from numpy import transpose
from numpy.linalg import inv
from .mueller import linear_retarder,linear_polarizer
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def condition_number(matrix):

    minv = np.linalg.pinv(matrix)

    # compute maximum norm https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    norm = np.linalg.norm(matrix,ord=np.inf)
    ninv = np.linalg.norm(minv,ord=np.inf)

    return norm * ninv 

def full_mueller_polarimetry(thetas,power=1,return_condition_number=False,Min=None):
    """conduct a full mueller polarimeter measurement

    Parameters
    ----------
    thetas : numpy.ndarray
        np.linspace(starting_angle,ending_angle,number_of_measurements)
    power : float, optional
        power recorded on a given pixel. Defaults to 1
    return_condition_number : bool, optional
        returns condition number of the data reduction matrix. by default False
    Min : numpy.ndarray
        if provided, is the "true" Mueller matrix. This allows us to
        simulate full mueller polarimetry. by default None

    Returns
    -------
    numpy.ndarray
        Mueller matrix measured by the polarimeter
    """
    nmeas = len(thetas)

    Wmat = np.zeros([nmeas,16])
    Pmat = np.zeros([nmeas])
    th = thetas

    for i in range(nmeas):

        # Mueller Matrix of Generator using a QWR
        Mg = linear_retarder(th[i],np.pi/2) @ linear_polarizer(0)

        # Mueller Matrix of Analyzer using a QWR
        Ma = linear_polarizer(0) @ linear_retarder(th[i]*5,np.pi/2)

        ## Mueller Matrix of System and Generator
        # The Data Reduction Matrix
        Wmat[i,:] = np.kron(Ma[0,:],Mg[:,0])

        # A detector measures the first row of the analyzer matrix and first column of the generator matrix
        if Min is not None:
            Pmat[i] = (Ma[0,:] @ Min @ Mg[:,0]) * power
        else:
            Pmat[i] = power[i]

    # Compute Mueller Matrix with Moore-Penrose Pseudo Inverse
    # Calculation appears to be sensitive to the method used to compute the inverse! There's something I guess
    M = np.linalg.pinv(Wmat) @ Pmat
    M = np.reshape(M,[4,4])

    if return_condition_number == True:
        return M,condition_number(Wmat)

    else:
        return M
    
