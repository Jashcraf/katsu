from __future__ import annotations

from typing import Any

import zodiax as zdx

from katsu.katsu_math import np
from katsu.mueller import linear_diattenuator, linear_retarder


class MuellerMatrix(zdx.Base):
    """Base class for all Mueller matrix elements.

    Can also be used to wrap an arbitrary 4x4 Mueller matrix. Provides support
    for Python's matrix multiplication operator (`@`) to allow composing
    multiple optical elements into a single composite `MuellerMatrix`.
    """

    matrix: Any
    n_params: int

    def __init__(self, matrix: Any, n_params: int = 0):
        """
        Parameters
        ----------
        matrix : np.ndarray
            The 4x4 Mueller matrix.
        n_params : int, optional
            The number of optimizeable parameters. Defaults to 0.
        """
        self.matrix = matrix
        self.n_params = n_params

    def matrix_from_params(self, p):
        """Returns the matrix computed from a vector of parameters.

        This is used by the `Interface` class during forward modeling. Fixed
        optics return their stored matrix, while variable optics override this
        method to recompute the matrix from the parameters `p`.
        """
        return self.matrix

    def __matmul__(self, other):
        if hasattr(other, "matrix"):
            return MuellerMatrix(self.matrix @ other.matrix)
        # Handle jax arrays, numpy arrays, or other array-likes safely
        elif hasattr(other, "shape") and len(getattr(other, "shape", ())) == 2:
            return MuellerMatrix(self.matrix @ other)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if hasattr(other, "shape") and len(getattr(other, "shape", ())) == 2:
            return MuellerMatrix(other @ self.matrix)
        else:
            return NotImplemented


class LinearRetarder(MuellerMatrix):
    """Homogeneous linear retarder.

    Parameters
    ----------
    fast_axis : float
        Fast-axis angle w.r.t. horizontal, in radians.
    retardance : float
        Retardance in radians.
    variable : bool or list[bool]
        Which parameters are free.  List order: ``[fast_axis, retardance]``.

    Examples
    --------
    Free retardance only:

    >>> LinearRetarder(fast_axis=0.0, retardance=np.pi / 2,
    ...                variable=[False, True])

    Both parameters free:

    >>> LinearRetarder(fast_axis=0.0, retardance=np.pi / 2, variable=True)
    """

    # Typing required by zodiax/equinox
    _fast_axis: float
    _retardance: float

    def __init__(self, fast_axis, retardance, variable=False):
        self._fast_axis = float(fast_axis)
        self._retardance = float(retardance)
        self.n_params = 2 if variable else 0
        self.matrix = linear_retarder(self._fast_axis, self._retardance)

    @property
    def params(self):
        return (self._fast_axis, self._retardance)

    def matrix_from_params(self, p):
        if self.n_params == 0:
            return self.matrix
        return linear_retarder(p[0], p[1])

    @classmethod
    def as_quarter_wave_plate(cls, fast_axis, variable=False):
        """Quarter-wave plate (retardance = π/2)."""
        return cls(fast_axis, np.pi / 2, variable=variable)

    @classmethod
    def as_half_wave_plate(cls, fast_axis, variable=False):
        """Half-wave plate (retardance = π)."""
        return cls(fast_axis, np.pi, variable=variable)


class LinearDiattenuator(MuellerMatrix):
    """Homogeneous linear diattenuator (partial or ideal polarizer).

    Parameters
    ----------
    transmission_axis : float
        Transmission-axis angle w.r.t. horizontal, in radians.
    Tmin : float
        Transmission of the blocked state (0 for an ideal polarizer).
    variable : bool or list[bool]
        Which parameters are free.
        List order: ``[transmission_axis, Tmin]``.
    """

    # Typing required by zodiax/equinox
    _transmission_axis: float
    _Tmin: float

    def __init__(self, transmission_axis, Tmin, variable=False):
        self._transmission_axis = float(transmission_axis)
        self._Tmin = float(Tmin)
        self.n_params = 2 if variable else 0
        self.matrix = linear_diattenuator(self._transmission_axis, self._Tmin)

    @property
    def params(self):
        return (self._transmission_axis, self._Tmin)

    def matrix_from_params(self, p):
        if self.n_params == 0:
            return self.matrix
        return linear_diattenuator(p[0], p[1])

    @classmethod
    def as_polarizer(cls, transmission_axis, variable=False):
        """Ideal linear polarizer (Tmin = 0)."""
        return cls(transmission_axis, 0.0, variable=variable)


class Model(zdx.Base):
    """Model fitting interface for polarimetry.

    Parameters
    ----------
    optics_list : list of MuellerMatrix
        The sequence of optical elements in the model.
    data : np.ndarray, optional
        Observed data to compare against in the forward model.
    """

    optics: list[MuellerMatrix]
    data: np.ndarray
    n_params: int
    offsets: list[tuple[int, int]]
    _fg_func: callable or None

    def __init__(self, optics_list, data=None):
        self.optics = optics_list
        self.data = data
        self.n_params = 0

        # Calculate offsets for the flat parameter vector x
        self.offsets = []
        assert any([optic.variable for optic in self.optics]), (
            "No variable optics found, class initialization failed"
        )
        cursor = 0
        for optic in self.optics:
            self.offsets.append((cursor, cursor + optic.n_params))
            cursor += optic.n_params
            self.n_params += optic.n_params

        self._fg_func = None

    def forward(self, x, stokes=None):
        """Compute the forward model / objective function.

        Parameters
        ----------
        x : np.ndarray
            The vector of all optimizeable parameters.
        stokes : np.ndarray, optional
            The input Stokes vector. Defaults to [1, 0, 0, 0].
        """
        if stokes is None:
            stokes = np.array([1.0, 0.0, 0.0, 0.0])

        system_matrix = np.eye(4)
        for optic, offset in zip(self.optics, self.offsets):
            if offset is not None:
                p = x[offset[0] : offset[1]]
                # Optics are applied in sequence (M_n @ ... @ M_1)
                system_matrix = optic.matrix_from_params(p) @ system_matrix
            else:
                system_matrix = optic.matrix @ system_matrix

        final_stokes = system_matrix @ stokes

        if self.data is not None:
            # Simple MSE loss if data is provided
            return np.mean((final_stokes - self.data) ** 2)

        return final_stokes[0]  # Return intensity by default

    def fg(self, x):
        """Compute both the function value and its gradient.

        Parameters
        ----------
        x : np.ndarray
            The vector of all optimizeable parameters.
        """
        if self._fg_func is None:
            try:
                import jax

                self._fg_func = jax.value_and_grad(self.forward)
            except ImportError:
                raise ImportError(
                    "JAX is required for gradient computation. "
                    "Please install it or set the katsu backend to JAX."
                )

        return self._fg_func(x)
