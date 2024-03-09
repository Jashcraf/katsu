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
    """Convenience method to automatically configure poke's backend to cupy."""
    import numpy as cp

    np._srcmodule = cp

    return


def set_backend_to_cupy():
    """Convenience method to automatically configure poke's backend to cupy."""
    import cupy as cp

    np._srcmodule = cp

    return

def broadcast_kron(a,b):
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

    return np.einsum('...ik,...jl',a,b).reshape([*a.shape[:-2],int(a.shape[-2]*b.shape[-2]),int(a.shape[-1]*b.shape[-1])])

def broadcast_outer(a,b):
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

    return np.einsum('...i,...j->...ij',a,b)