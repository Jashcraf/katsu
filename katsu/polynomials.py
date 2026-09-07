"""
Minor modifications of select prysm.polynomials functions to support gradient backpropagation with Jax. This module makes modifications to
`prysm.polynomials.zernike.zernike_nm_seq`, and `prysm.polynomials.jacobi.jacobi_seq`

prysm is open-source via the MIT License, and prysm is officially a dependency for Katsu to use spatial calibration
"""

# Prysm is now an official dependency
from collections import defaultdict

import numpy as truenp
from prysm.mathops import is_odd, kronecker, sign
from prysm.polynomials.jacobi import jacobi, jacobi_der, recurrence_abc
from prysm.polynomials.zernike import zernike_norm

from katsu.katsu_math import np


def zernike_nm_seq(nms, r, t, norm=True):
    """Zernike polynomial of radial order n, azimuthal order m at point r, t.

    Parameters
    ----------
    nms : iterable of tuple of int,
        seq of (n, m); looks like [(1,1), (3,1), ...]
    r : ndarray
        radial coordinates
    t : ndarray
        azimuthal coordinates
    norm : bool, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

    Returns
    -------
    ndarray
        shape (k, n, m), with k = len(nms)

    """
    # this function deduplicates all possible work.  It uses a connection
    # to the jacobi polynomials to efficiently compute a series of zernike
    # polynomials
    # it follows this basic algorithm:
    # for each (n, m) compute the appropriate Jacobi polynomial order
    # collate the unique values of that for each |m|
    # compute a set of jacobi polynomials for each |m|
    # compute r^|m| , sin(|m|*t), and cos(|m|*t for each |m|
    #
    # benchmarked at 12.26 ns/element (256x256), 4.6GHz CPU = 56 clocks per element
    # ~36% faster than previous impl (12ms => 8.84 ms)
    x = 2 * r**2 - 1
    ms = [e[1] for e in nms]
    am = truenp.abs(ms)
    amu = truenp.unique(am)

    def factory():
        return 0

    jacobi_seqs_mjn = defaultdict(factory)
    # jacobi_seqs_mjn is a lookup table from |m| to all orders < max(n_j)
    # for each |m|, i.e. 0 .. n_j_max
    for nm, am_ in zip(nms, am):
        n = nm[0]
        nj = (n - am_) // 2
        if nj > jacobi_seqs_mjn[am_]:
            jacobi_seqs_mjn[am_] = nj

    for k in jacobi_seqs_mjn:
        nj = jacobi_seqs_mjn[k]
        jacobi_seqs_mjn[k] = truenp.arange(nj + 1)

    jacobi_seqs = {}

    jacobi_seqs_mjn = dict(jacobi_seqs_mjn)
    for k in jacobi_seqs_mjn:
        n_jac = jacobi_seqs_mjn[k]
        jacobi_seqs[k] = list(jacobi_seq(n_jac, 0, k, x))

    powers_of_m = {}
    sines = {}
    cosines = {}
    for m in amu:
        powers_of_m[m] = r**m
        sines[m] = np.sin(m * t)
        cosines[m] = np.cos(m * t)

    out = np.empty((len(nms), *r.shape), dtype=r.dtype)
    k = 0
    for n, m in nms:
        absm = abs(m)
        nj = (n - absm) // 2
        jac = jacobi_seqs[absm][nj]
        if norm:
            jac = jac * zernike_norm(n, m)

        if m == 0:
            # rotationally symmetric Zernikes are jacobi
            if np.__name__ == "jax.numpy":
                out = out.at[k].set(jac)
            else:
                out[k] = jac

            k += 1
        else:
            if m < 0:
                azpiece = sines[absm]
            else:
                azpiece = cosines[absm]

            radialpiece = powers_of_m[absm]
            zern = jac * azpiece * radialpiece  # jac already contains the norm
            if np.__name__ == "jax.numpy":
                out = out.at[k].set(zern)
            else:
                out[k] = zern
            k += 1

    return out


def jacobi_seq(ns, alpha, beta, x):
    """Jacobi polynomials of orders ns with weight parameters alpha and beta.

    Parameters
    ----------
    ns : iterable
        sorted polynomial orders to return, e.g. [1, 3, 5, 7, ...]
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : ndarray
        x coordinates to evaluate at

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # previously returned a gnerator; ergonomics were not-good
    # typical usage would be array(list(jacobi_seq(...))
    # generator lowers peak memory consumption by allowing caller
    # to do weighted sums 'inline', but
    # for example (1024, 1024) x is ~8 megabytes per mode;
    # need to be in an edge case scenario for it to matter,
    # just return array for ergonomics
    ns = list(ns)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    if ns[min_i] == 0:
        if np.__name__ == "jax.numpy":
            out = out.at[min_i].set(1)
        else:
            out[min_i] = 1
        min_i += 1

    if min_i == len(ns):
        return out

    Pn = alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)
    if ns[min_i] == 1:
        if np.__name__ == "jax.numpy":
            out = out.at[min_i].set(Pn)
        else:
            out[min_i] = Pn
        min_i += 1

    if min_i == len(ns):
        return out

    Pnm1 = Pn
    A, B, C = recurrence_abc(1, alpha, beta)
    Pn = (A * x + B) * Pnm1 - C  # no C * Pnm2 =because Pnm2 = 1
    if ns[min_i] == 2:
        if np.__name__ == "jax.numpy":
            out = out.at[min_i].set(Pn)
        else:
            out[min_i] = Pn
        min_i += 1

    if min_i == len(ns):
        return out

    max_n = ns[-1]
    for i in range(3, max_n + 1):
        Pnm2, Pnm1 = Pnm1, Pn
        A, B, C = recurrence_abc(i - 1, alpha, beta)
        Pn = (A * x + B) * Pnm1 - C * Pnm2
        if ns[min_i] == i:
            if np.__name__ == "jax.numpy":
                out = out.at[min_i].set(Pn)
            else:
                out[min_i] = Pn
            min_i += 1

    return out
