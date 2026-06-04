from typing import Any

import zodiax as zdx

from katsu.katsu_math import np
from katsu.mueller import linear_diattenuator, linear_retarder


def _digest_variables(variable, params):
    """
    Digest the variable specification into a dictionary of parameters to vary.

    Parameters
    ----------
    variable : bool, list, or dict
        The variable specification.
    params : list
        The list of parameters to vary.

    Returns
    -------
    dict
        A dictionary of parameters to vary, with parameter names as keys and
        boolean values indicating whether to vary the parameter.
    """
    if isinstance(variable, bool):
        return {p: variable for p in params}
    elif isinstance(variable, list):
        return {p: True for p in params if p in variable}
    elif isinstance(variable, dict):
        return variable
    else:
        raise ValueError(f"Invalid variable type: {type(variable)}")


class MuellerMatrix(zdx.Base):
    """Base class for all Mueller matrix elements.

    Can also be used to wrap an arbitrary 4x4 Mueller matrix. Provides support
    for Python's matrix multiplication operator (`@`) to allow composing
    multiple optical elements into a single composite `MuellerMatrix`.

    Generally variable elements should be specified as a boolean, list, or dict.
    boolean: True means vary all free parameters, False means fix all free parameters
    dict: dict of attributes to vary and explicitly specify if they are variable

    internally, these are stored as a dictionary
    """

    matrix: Any
    variable: bool or dict
    shape: tuple or None
    _n_params: int
    _trainable: dict

    def __init__(self, matrix, variable=False, shape=None):
        """
        Parameters
        ----------
        matrix : np.ndarray
            The 4x4 Mueller matrix.
        n_params : int, optional
            The number of optimizeable parameters. Defaults to 0.
        """
        self.matrix = matrix
        self.variable = variable
        self.shape = shape
        self._n_params = 0
        self._trainable = {}

    def __matmul__(self, other):
        if hasattr(other, "matrix"):
            return MuellerMatrix(self.matrix @ other.matrix)

        # Handle jax arrays, numpy arrays, or other array-likes safely
        # This will return a raw array, not a `MuellerMatrix`, so the type
        # of the array to the right is maintained
        elif hasattr(other, "shape") and len(getattr(other, "shape", ())) >= 1:
            return self.matrix @ other

        else:
            return NotImplemented, "Unsupported operand type: " + str(type(other))

    def __rmatmul__(self, other):
        if hasattr(other, "shape") and len(getattr(other, "shape", ())) >= 1:
            return other @ self.matrix
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
    fast_axis: float or np.ndarray
    retardance: float or np.ndarray

    def __init__(self, fast_axis, retardance, variable=False, shape=None):

        # These can be floats or ndarrays because katsu accounts for the difference in shape
        self.fast_axis = fast_axis
        self.retardance = retardance

        # From MuellerMatrix
        self.matrix = linear_retarder(self.fast_axis, self.retardance, shape=shape)
        self.shape = shape

        # Digest trainable parameters
        self.variable = variable
        self._trainable = _digest_variables(self.variable, ["fast_axis", "retardance"])
        self._n_params = sum(self._trainable.values())

    @classmethod
    def as_quarter_wave_plate(cls, fast_axis, variable=False, shape=None):
        """Quarter-wave plate (retardance = π/2). This also sets retardance to π/2, and
        does not permit Tmin to be varied."""
        if not isinstance(variable, bool):
            if variable["retardance"]:
                raise ValueError(
                    "Retardance cannot be varied in a quarter-wave plate model"
                )
        else:
            variable = {"fast_axis": variable}

        return cls(
            fast_axis=fast_axis, retardance=np.pi / 2, variable=variable, shape=shape
        )

    @classmethod
    def as_half_wave_plate(cls, fast_axis, variable=False, shape=None):
        """Half-wave plate (retardance = π). This also sets retardance to π, and
        does not permit Tmin to be varied."""
        if not isinstance(variable, bool):
            if variable["retardance"]:
                raise ValueError(
                    "Retardance cannot be varied in a half-wave plate model"
                )
        else:
            variable = {"fast_axis": variable}

        return cls(
            fast_axis=fast_axis, retardance=np.pi, variable=variable, shape=shape
        )

    def update(self, x):
        """
        Update the model parameters and matrix from the given vector `x`.

        This accepts a vector of parameters whose length must match the number
        of trainable parameters (i.e. 2 for a linear retarder).

        Parameters
        ----------
        x : array-like
            The vector of parameters to update the model with. This is ordered
            as fast axis, then retardance.
        """
        if len(x) != len(self._trainable):
            raise ValueError(
                f"Expected {len(self._trainable)} parameters, got {len(x)}"
            )

        # Update free parameters if trainable
        for i, param in enumerate(self._trainable):
            if self._trainable[param]:
                setattr(self, param, x[i])

        # Update the Mueller matrix
        self.matrix = linear_retarder(self.fast_axis, self.retardance, shape=self.shape)
        return self.matrix


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
    transmission_axis: float
    Tmin: float
    _n_params: int

    def __init__(self, transmission_axis, Tmin, variable=False, shape=None):
        self.transmission_axis = float(transmission_axis)
        self.Tmin = float(Tmin)

        # from MuellerMatrix
        self.matrix = linear_diattenuator(
            self.transmission_axis, self.Tmin, shape=shape
        )
        self.shape = shape

        # Digest trainable parameters
        self.variable = variable
        self._trainable = _digest_variables(
            self.variable, ["transmission_axis", "Tmin"]
        )
        self._n_params = sum(self._trainable.values())

    @classmethod
    def as_polarizer(cls, transmission_axis, variable=False, shape=None):
        """Ideal linear polarizer (Tmin = 0). This also sets Tmin to zero, and
        does not permit Tmin to be varied."""
        if not isinstance(variable, bool):
            if variable["Tmin"]:
                raise ValueError("Tmin cannot be varied in a polarizer model")
        else:
            variable = {"transmission_axis": variable}

        return cls(transmission_axis, 0.0, variable=variable, shape=shape)

    def update(self, x):
        """
        Update the model parameters and matrix from the given vector `x`.

        Parameters
        ----------
        x : array-like
            The vector of parameters to update the model with.
        """

        # Update free parameters if trainable
        for i, param in enumerate(self._trainable):
            if self._trainable[param]:
                setattr(self, param, x[i])

        # Update the Mueller matrix
        self.matrix = linear_diattenuator(
            self.fast_axis, self.retardance, shape=self.shape
        )
        return self.matrix


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
    offsets: list[tuple[int, int]]
    system_matrix: np.ndarray
    _n_params: int
    _fg_func: callable or None

    def __init__(self, optics_list, data=None):
        self.optics = optics_list
        self.data = data
        self._n_params = 0

        # Calculate offsets for the flat parameter vector x
        self.offsets = []
        cursor = 0

        # Build understanding of slices for parameter vector x
        for optic in self.optics:
            self.offsets.append((cursor, cursor + optic._n_params))
            cursor += optic._n_params
            self._n_params += optic._n_params

        self._fg_func = None
        self.system_matrix = np.eye(4)

    def _update(self, x):
        """Update the optics and full system matrix

        Parameters
        ----------
        x : np.ndarray
            The vector of all optimizeable parameters.

        """
        cursor = 0
        system_matrix = np.eye(4)

        for optic, offsets in zip(self.optics, self.offsets):
            # Update optic variables
            if optic._n_params > 0:
                optic.update(x[cursor : cursor + optic._n_params])
            cursor += optic._n_params

            # Update the system matrix
            system_matrix = optic.matrix @ system_matrix

        self.system_matrix = system_matrix

    def forward(self, x, stokes=None):
        """Compute the forward model / objective function.

        Parameters
        ----------
        x : np.ndarray
            The vector of all optimizeable parameters.
        stokes : np.ndarray, optional
            The input Stokes vector. Defaults to [1, 0, 0, 0].

        """
        # Update the optics
        self._update(x)

        if stokes is None:
            stokes = np.array([1.0, 0.0, 0.0, 0.0])

        # Multiply by stokes
        final_stokes = system_matrix @ stokes

        return final_stokes[..., 0]  # Return intensity observed

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
