from katsu.katsu_math import np

DRRP_PARAMS = [
    "PSG_WVP_ANG",
    "PSG_WVP_RET",
    "PSG_POL_ANG",
    "PSG_POL_DIA",
    "PSA_WVP_ANG",
    "PSA_WVP_RET",
    "PSA_POL_ANG",
    "PSA_POL_DIA",
]


class DualRotatingRetarderPolarimeter:
    def __init__(
        self,
        free_params: list[str],
        fixed_params: dict,
        psg_angles: np.ndarray,
        psa_angles: np.ndarray,
    ):

        # Handle duplicates, permits logical operations between dtypes
        covered = set(fixed_params.keys()) | set(free_params)
        missing = set(DRRP_PARAMS) - covered
        overlap = set(fixed_params.keys()) & set(free_params)

        if missing:
            raise ValueError(f"Parameters not assigned (fixed or free): {missing}")
        if overlap:
            raise ValueError(f"Parameters appear in both fixed and free: {overlap}")

        self.free_params = free_params
        self.fixed_params = fixed_params
        self.psg_angles = psg_angles
        self.psa_anlges = psa_angles

    @classmethod
    def from_rotation_ratio(
        cls,
        free_params: list[str],
        fixed_params: dict,
        psg_angles: np.ndarray,
        rotation_ratio: float,
    ):

        psa_angles = psg_angles * rotation_ratio

        return cls(free_params, fixed_params, psg_angles, psa_angles)

    def build_forward_model(self):

        # Builds forward model that calls free_params


class DualRotatingRetarder:
    def __init__(
        self,
        psg_angles,
        psa_angles,
        psg_retardance,
        psa_retardance,
        psg_diattenuation,
        psa_diattenuation,
        psg_wvp_angle,
        psa_wvp_angle,
        psg_pol_angle,
        psa_pol_angle,
        basis=None,
    ):
        """
        Parameters
        ----------
        psg_angles: ndarray
            array of command angles sent to the psg retarder
        psa_angles: ndarray
            array of command angles sent to the psa retarder
        psg_retardance: float or ndarray
            retardance of the psg waveplate
        psa_retardance: float or ndarray
            retardance of the psa waveplate
        psg_diattenuation: float or ndarray
            diattenuation of the psg polarizer
        psa_diattenuation: float or ndarray
            diattenuation of the psa polarizer
        psg_wvp_angle: float or ndarray
            starting angle of the psg waveplate
        psa_wvp_angle: float or ndarray
            starting angle of the psa waveplate
        psg_pol_angle: float or ndarray
            angle of the psg polarizer
        psa_pol_angle: float or ndarray
            angle of the psa polarizer
        basis: array or list
            containing the shapes that array-typed free-parameters
            will be computed with
        """

        # Distill model parameters into generally spatial modes
        pass
