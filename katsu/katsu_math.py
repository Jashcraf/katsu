import numpy as np

class BackendShim:
    """A shim that allows a backend to be swapped at runtime.
    Taken from prysm.mathops with permission from Brandon Dube
    """

    def __init__(self, src):
        self._srcmodule = src
 
    def __getattr__(self, key):
        if key == "_srcmodule":
            return self._srcmodule

        return getattr(self._srcmodule, key)


_np = np
np = BackendShim(_np)


def set_backend_to_numpy():
    """Convenience method to automatically configure katsu's backend to numpy."""
    import numpy

    np._srcmodule = numpy

    return


def set_backend_to_cupy():
    """Convenience method to automatically configure katsu's backend to cupy."""
    import cupy as cp

    np._srcmodule = cp

    return

def set_backend_to_jax(): 
    """Convenience method to automatically configure katsu's backend to jax."""
    import jax as jax

    jax.config.update("jax_enable_x64", True)
    np._srcmodule = jax.numpy

    return

def broadcast_kron(a, b):
    """broadcasted kronecker product of two N,M,...,2,2 arrays. Used for jones -> mueller conversion
    In the unbroadcasted case, this output looks like

    out = [a[0,0]*b,a[0,1]*b]
          [a[1,0]*b,a[1,1]*b]

    where out is a N,M,...,4,4 array. I wrote this to work for generally shaped kronecker products where the matrix
    is contained in the last two axes, but it's only tested for the Nx2x2 case

    Parameters
    ----------
    a : numpy.ndarray
        N,M,...,2,2 array used to scale b in kronecker product
    b : numpy.ndarray
        N,M,...,2,2 array used to form block matrices in kronecker product

    Returns
    -------
    out
        N,M,...,4,4 array
    """

    return np.einsum('...ik,...jl', a, b).reshape([*a.shape[:-2],int(a.shape[-2]*b.shape[-2]),int(a.shape[-1]*b.shape[-1])])


def broadcast_outer(a, b):
    """broadcasted outer product of two A,B,...,N vectors. Used for polarimetric data reduction

    where out is a A,B,...,N,N matrix. While in principle this does not require vectors of different length, it is not tested
    to produce anything other than square matrices.

    Parameters
    ----------
    a : numpy.ndarray
        A,B,...,N vector 1
    b : numpy.ndarray
        A,B,...,N vector 2

    Returns
    -------
    numpy.ndarray
        outer product matrix
    """

    return np.einsum('...i,...j->...ij', a, b)


def condition_number(matrix):
    """returns the condition number of a matrix. Useful for quantifying the quality
    of a polarimeter.

    Parameters
    ----------
    matrix : numpy.ndarray
        array containing the matrices to evaluate in the last two dimensions

    Returns
    -------
    numpy.ndarray
        condition number
    """

    minv = np.linalg.pinv(matrix)

    # compute maximum norm https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    norm = np.linalg.norm(matrix, ord=np.inf, axis=(-2,-1))
    ninv = np.linalg.norm(minv, ord=np.inf, axis=(-2,-1))

    return norm * ninv


M_identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def RMS_calculator(calibration_matrix):
    """Calculates the root mean square (RMS) error of a matrix by comparing it to the identity matrix.

    Parameters
    ----------
    calibration_matrix : array
        4x4 matrix.

    Returns
    -------
    RMS : float
        RMS error of the matrix."""
    differences = []
    for i in range(0, 4):
        for j in range(0, 4):
            differences.append(calibration_matrix[i, j]-M_identity[i, j])

    differences_squared = [x**2 for x in differences]
    RMS = np.sqrt(sum(differences_squared)/16)
    return RMS


# Calculate the retardance error by standard error propogation using RMS in the matrix elements from calibration
def propagated_error(M_R, RMS):
    """Propogates error in the Mueller matrix to error in the extracted value of retardance. 
    Assumes the RMS error is the same for all elements of the matrix.

    Parameters
    ----------
    M_R : array
        4x4 Mueller matrix for a linear retarder
    RMS : float
        root mean square error of the Mueller matrix.

    Returns
    ------- 
        float, error in the extracted retardance value in radians.
    """
    x = np.trace(M_R)
    return 2*RMS/np.sqrt(4*x-x**2) # Value in radians
