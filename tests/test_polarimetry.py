from katsu.katsu_math import np, broadcast_outer
from katsu.mueller import (
    linear_diattenuator,
    linear_retarder,
    linear_polarizer,
    stokes_from_parameters
)
from katsu.polarimetry import (
    full_mueller_polarimetry,
    stokes_sinusoid,
    full_stokes_polarimetry
)

NMEAS = 42
PSA_ANGULAR_INCREMENT = 5
thetas = np.linspace(0, np.pi, NMEAS)
rand_retarder_shaped = linear_retarder(np.random.random(),
                                       np.random.random(),
                                       shape=[32, 32, NMEAS])
rand_diattenuator_shaped = linear_diattenuator(np.random.random(),
                                               Tmin=np.random.random(),
                                               shape=[32, 32, NMEAS])
rand_M_shaped = rand_retarder_shaped @ rand_diattenuator_shaped
rand_M = rand_M_shaped[0, 0]


def test_full_mueller_polarimetry():

    # set up polarization state generator
    psg_polarizer = linear_polarizer(0, shape=thetas.shape)
    psg_retarder = linear_retarder(thetas, np.pi / 2, shape=thetas.shape)
    PSG = psg_retarder @ psg_polarizer

    # set up polarization state generator
    psa_polarizer = linear_polarizer(0, shape=thetas.shape)
    psa_retarder = linear_retarder(PSA_ANGULAR_INCREMENT * thetas, np.pi / 2, shape=thetas.shape)
    PSA = psa_polarizer @ psa_retarder

    # set up system Mueller matrix
    Msys = PSA @ rand_M @ PSG

    # propagate Stokes vector to get power
    Sin = stokes_from_parameters(1, 0, 0, 0)
    Sout = Msys @ Sin
    power_measured = Sout[..., 0, 0]

    M_measured = full_mueller_polarimetry(thetas, power_measured, PSA_ANGULAR_INCREMENT)

    np.testing.assert_allclose(M_measured, rand_M[0], rtol=1e-5, atol=1e-7)


def test_full_mueller_polarimetry_broadcast():

    # set up polarization state generator
    psg_polarizer = linear_polarizer(0, shape=[32, 32, NMEAS])
    psg_retarder = linear_retarder(thetas, np.pi / 2, shape=[32, 32, NMEAS])
    PSG = psg_retarder @ psg_polarizer

    # set up polarization state generator
    psa_polarizer = linear_polarizer(0, shape=[32, 32, NMEAS])
    psa_retarder = linear_retarder(PSA_ANGULAR_INCREMENT * thetas, np.pi / 2, shape=[32, 32, NMEAS])
    PSA = psa_polarizer @ psa_retarder

    # set up system Mueller matrix
    Msys = PSA @ rand_M_shaped @ PSG

    # propagate Stokes vector to get power
    Sin = stokes_from_parameters(1, 0, 0, 0)
    Sout = Msys @ Sin
    power_measured = Sout[..., 0, 0]

    M_measured = full_mueller_polarimetry(thetas, power_measured, PSA_ANGULAR_INCREMENT)

    np.testing.assert_allclose(M_measured, rand_M_shaped[..., 0, :, :], rtol=1e-5, atol=1e-7)


def test_stokes_sinusoid():
    a0 = 1
    b2 = -1
    a4 = 0.5
    b4 = 0.25
    theta = np.linspace(0, 10*np.pi)
    sinusoid = a0 + b2*np.sin(2*theta) + a4*np.cos(4*theta) + b4*np.sin(4*theta)

    np.testing.assert_allclose(stokes_sinusoid(theta, a0, b2, a4, b4), sinusoid)


def test_full_stokes_polarimetry():
    thetas = np.linspace(0, np.pi, 10)
    S_to_measure = np.random.random(4)
    power_matrix = np.zeros_like(thetas)

    for i, angle in enumerate(thetas):
        PSA = linear_polarizer(0) @ linear_retarder(angle, np.pi/2)
        power_matrix[i] = (PSA[0, :] @ S_to_measure)

    S_out = full_stokes_polarimetry(thetas, power=power_matrix)

    np.testing.assert_allclose(S_to_measure, S_out)










