from typing import Any

import zodiax as zdx

from katsu.katsu_math import np
from katsu.mueller import linear_diattenuator, linear_retarder
from katsu.helpers import list2dictionary
from collections import OrderedDict

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
    """Base class for differentiable Mueller matrix elements.

    Provides support for Python's matrix multiplication operator (`@`) to allow
    composing multiple optical elements into a single composite `MuellerMatrix`.

    Parametric subclasses (e.g. ``LinearRetarder``) store only the *physical
    parameters* (e.g. ``fast_axis``, ``retardance``) as pytree leaves and expose
    ``matrix`` as a computed ``@property``. The matrix is therefore rebuilt from
    the current parameters every time it is accessed, which is what makes these
    objects play nicely with zodiax: a ``model.set("retarder.fast_axis", value)``
    swaps the parameter leaf and the next access of ``.matrix`` automatically
    reflects the update. (If ``matrix`` were a stored leaf, ``.set`` on a
    parameter would leave the cached matrix stale.)

    A ``MuellerMatrix`` can also be constructed directly from an already-computed
    matrix. This is what ``@`` returns when two elements are composed, so the
    product of two optics is itself a ``MuellerMatrix`` (and can be composed
    further or have its ``.matrix`` read). In that case the matrix is a stored
    leaf rather than a computed property.
    """

    _matrix: Any = None

    def __init__(self, matrix=None):
        # A directly-constructed MuellerMatrix (e.g. the product of two
        # elements) stores its precomputed matrix here. Parametric subclasses
        # leave this None and override the `matrix` property instead.
        self._matrix = matrix

    @property
    def matrix(self):
        if self._matrix is None:
            raise NotImplementedError(
                "Subclasses of MuellerMatrix must implement the `matrix` property."
            )
        return self._matrix

    def __matmul__(self, other):
        if hasattr(other, "matrix"):
            return MuellerMatrix(self.matrix @ other.matrix)

        # Handle jax arrays, numpy arrays, or other array-likes safely
        # This will return a raw array, not a `MuellerMatrix`, so the type
        # of the array to the right is maintained
        elif hasattr(other, "shape") and len(getattr(other, "shape", ())) >= 1:
            return self.matrix @ other

        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if hasattr(other, "shape") and len(getattr(other, "shape", ())) >= 1:
            return other @ self.matrix
        else:
            return NotImplemented


class LinearDiattenuator(MuellerMatrix):
    # Typing required by zodiax/equinox. Only the physical parameters are
    # leaves; `matrix` is computed from them (see the property below).
    transmission_axis: float or np.ndarray
    Tmin: float or np.ndarray
    shape: tuple or None
    offset: float

    def __init__(self, transmission_axis, Tmin, shape=None, offset=0.):
        # Do NOT cast to float(): keep values as-is so they stay traceable
        # under jax (jit/grad) and so array-valued parameters are supported.
        self.transmission_axis = transmission_axis
        self.Tmin = Tmin
        self.shape = shape
        self.offset = offset

    @property
    def matrix(self):
        return linear_diattenuator(
            self.transmission_axis + self.offset, self.Tmin, shape=self.shape
        )

    @classmethod
    def as_polarizer(cls, transmission_axis, shape=None):
        """Ideal linear polarizer: a diattenuator with ``Tmin = 0``."""
        return cls(transmission_axis, 0.0, shape=shape)


class LinearRetarder(MuellerMatrix):
    # Typing required by zodiax/equinox. Only the physical parameters are
    # leaves; `matrix` is computed from them (see the property below).
    fast_axis: float or np.ndarray
    retardance: float or np.ndarray
    offset: float
    shape: tuple or None

    def __init__(self, fast_axis, retardance, shape=None, offset=0.):
        # These can be floats or ndarrays because katsu accounts for the
        # difference in shape.
        self.offset = offset
        self.fast_axis = fast_axis
        self.retardance = retardance
        self.shape = shape

    @property
    def matrix(self):
        # `offset` is applied here rather than folded into `fast_axis` at
        # construction. Baking it in would make the axis a stored leaf that no
        # longer tracks `offset`, so `.set("...offset", x)` would silently
        # leave the matrix stale and hand back a zero gradient.
        return linear_retarder(
            self.fast_axis + self.offset, self.retardance, shape=self.shape
        )

    @classmethod
    def as_quarter_wave_plate(cls, fast_axis, shape=None):
        """Quarter-wave plate: a linear retarder with retardance ``pi / 2``."""
        return cls(fast_axis, np.pi / 2, shape=shape)

    @classmethod
    def as_half_wave_plate(cls, fast_axis, shape=None):
        """Half-wave plate: a linear retarder with retardance ``pi``."""
        return cls(fast_axis, np.pi, shape=shape)


class Model(zdx.Base):
    layers: OrderedDict

    def __init__(self, layers):

        # Digest the (name, optic) list into an ordered dict of layers
        self.layers = list2dictionary(layers, ordered=True)

    def __getattr__(self, key):
        if key in self.layers.keys():
            return self.layers[key]
        
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)

        raise AttributeError(f"Model has no attribute '{key}'")

    # Does forward modeling things
    def forward(self, stokes=np.array([1., 0., 0., 0.])):
        system_matrix = np.eye(4)
        for layer in list(self.layers.values()):
            system_matrix = layer @ system_matrix

        return system_matrix @ stokes


