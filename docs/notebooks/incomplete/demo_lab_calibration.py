import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as tnp
import time
import sys
from pathlib import Path

# Less common imports
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.polynomials import noll_to_nm, sum_of_2d_modes
from prysm.polynomials import zernike_nm_sequence
from prysm.geometry import circle

from katsu.mueller import (
    linear_diattenuator,
    linear_retarder,
    linear_polarizer,
    retardance_from_mueller
)

from katsu.polarimetry import drrp_data_reduction_matrix
from katsu.katsu_math import np, set_backend_to_jax, broadcast_kron

# derpy polarimeter libraries
sys.path.append("/Users/kaladin/derp_control/")
from derpy.writing import read_experiment

DATA_PTH = Path.home() / "Library/CloudStorage/Box-Box/97_Data/derp/20250210_GPI"

CALIBRATION_ID = DATA_PTH / "GPI_HWP_center_air_calibration.msgpack"
EXPERIMENT_ID = DATA_PTH / "GPI_HWP_center_measure_0.msgpack"

# TODO: Make these not redundant
WVL_ID = 4
WAVELENGTH_SELECT = 1500  # nm

PLOT_INTERMEDIATE = True
MASK_RAD = 0.5 # from 0 to 1, 1 being the full circle
MODE = "both" # "left", "right", "both"
BAD_FRAMES_CAL = [] # [3, 13 -1], _, [8]
BAD_FRAMES = [4]  # [-9], _, [-11]
PLOT_IMAGES = False
# --------------------------------


# Plotting stuff
plt.style.use('bmh')
okabe_colorblind8 = ['#E69F00','#56B4E9','#009E73',
                     '#F0E442','#0072B2','#D55E00',
                     '#CC79A7','#000000']

plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=okabe_colorblind8)

def arccos_taylor(x):
    """
    Taylor series centered on -1

    Parameters
    ----------
    x : ndarray
        input array of floats
    """

    t1 = np.pi / 2
    t2 = x
    t3 = x**3 / 6
    t4 = 3 * x**5 / 40

    return t1 - t2 - t3 - t4 # - t5 - t6 - t7

def retardance_from_mueller_taylor(x):
    """Computes the retardance from a Mueller matrix using the
    Taylor series version of the arccos function

    Parameters
    ----------
    x : ndarray
        Mueller matrix of shape (N, N, 4, 4)

    Returns
    -------
    retardance : ndarray
        retardance of shape (N, N)
    """

    tracem = np.trace(x, axis1=-1, axis2=-2) / 2
    retardance = arccos_taylor(tracem - 1)
    return retardance


def plot_square(x,n=4,vmin=None,vmax=None, common_cbar=True):
    """A simple plotting macro for viewing intermediate results
    """
    k = 1
    plt.figure(figsize=[10,10])
    for i in range(n):
        for j in range(n):
            plt.subplot(n,n,k)
            im = plt.imshow(x[..., i, j], vmin=vmin, vmax=vmax, cmap='RdBu_r')
            if not common_cbar:
                plt.colorbar()
            k += 1

    if common_cbar:
        ax = plt.gca()
        cbar = plt.colorbar(im, ax=ax)

    # plt.show()


def jax_sum_of_2d_modes(modes, weights):
    """a clone of prysm.polynomials sum_of_2d_modes that works when using katsu's Jax backend

    Parameters
    ----------
    modes : list of ndarrays
        list of polynomials constituting the desired basis
    weights : list of floats
        coefficients that describe the presence of each mode in the modes list

    Returns
    -------
    ndarray
        2D ndarray containing the sum of weighted modes
    """
    modes = np.asarray(modes)
    weights = np.asarray(weights).astype(modes.dtype)

    # dot product of the 0th dim of modes and weights => weighted sum
    return np.tensordot(modes, weights, axes=(0, 0))

def sum_of_2d_modes_wrapper(modes, weights):
    """ Wrapper that lets us ignore which source module we want to use
    """
    if np._srcmodule == tnp:
        return sum_of_2d_modes(modes, weights)
    else:
        return jax_sum_of_2d_modes(modes, weights)


def construct_zern_basis(r, t, NMODES):
    """Builds a prysm zernike basis

    Parameters
    ----------
    r : ndarray
        radial coordinate
    t : ndarray
        azimuthal coordinate
    NMODES: int
        Maximum zernike noll index to simulate

    Returns
    -------
    list
        list whose elements contain Zernike modes
    """

    nms = [noll_to_nm(i) for i in range(1, NMODES)]

    # Norm = False is required to have unit peak-to-valley
    basis_full = list(zernike_nm_sequence(nms, r, t, norm=False))

    A = circle(1, r) # a circular mask to apply to the beam
    basis = [mode * A for mode in basis_full ]

    return basis

def rotation_matrix(th):
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th), np.cos(th)]])

def forward_simulate(x, NMODES, NMEAS):
    """Simulates a polarimetric measurement with
    spatially-varying data

    Parameters
    ----------
    x : ndarray or list
        parameter input vector
    NMODES: int
        Maximum zernike noll index to simulate
    NMEAS: int
        Number of angular measurements to perform

    Returns
    -------
    ndarray
        Power observed of shape NPIX X NPIX X NMEAS
    """

    ROTATION_RATIO = 2.5
    END_ANGLE_PSG = 180


    # unpack the parameters
    theta_pg = x[0] # Starting angle of the polarizer
    theta_pa = x[1] # Starting angle of the polarizer

    # get spatially varying coefficcients
    coeffs_spatial_ret_psg = x[2 : 2 + 1*len(basis)]
    coeffs_spatial_ang_psg = x[2 + 1*len(basis):(2 + 2*len(basis))]
    coeffs_spatial_ret_psa = x[2 + 2*len(basis):(2 + 3*len(basis))]
    coeffs_spatial_ang_psa = x[2 + 3*len(basis):(2 + 4*len(basis))]

    # set up the basis with rotation
    x, y = make_xy_grid(NPIX, diameter=2)
    r, t = cart_to_polar(x, y)

    # the nominal rotations performed
    rotations_psg = np.linspace(0, np.radians(END_ANGLE_PSG), NMEAS)
    rotations_psa = rotations_psg * ROTATION_RATIO

    psg_retardances, psa_retardances = [], []
    psg_fast_axes, psa_fast_axes = [], []

    for rot_psg, rot_psa in zip(rotations_psg, rotations_psa):

        # Get the rotated spatial basis
        basis_psg = np.asarray(construct_zern_basis(r, t + rot_psg, NMODES+1))
        basis_psa = np.asarray(construct_zern_basis(r, t + rot_psa, NMODES+1))

        # compute retardances
        psg_retardance = sum_of_2d_modes_wrapper(basis_psg, coeffs_spatial_ret_psg)
        psa_retardance = sum_of_2d_modes_wrapper(basis_psa, coeffs_spatial_ret_psa)

        # compute fast axes
        psg_fast_axis = sum_of_2d_modes_wrapper(basis_psg, coeffs_spatial_ang_psg)
        psa_fast_axis = sum_of_2d_modes_wrapper(basis_psa, coeffs_spatial_ang_psa)

        # store arrays in list
        psg_retardances.append(psg_retardance)
        psa_retardances.append(psa_retardance)
        psg_fast_axes.append(psg_fast_axis)
        psa_fast_axes.append(psa_fast_axis)

    # get lists as arrays
    psg_retardances = np.asarray(psg_retardances)
    psa_retardances = np.asarray(psa_retardances)
    psg_fast_axes = np.asarray(psg_fast_axes) + rotations_psg[..., None, None]
    psa_fast_axes = np.asarray(psa_fast_axes) + rotations_psa[..., None, None]

    # swap axes around
    psg_retardances = np.moveaxis(psg_retardances, 0, -1)
    psa_retardances = np.moveaxis(psa_retardances, 0, -1)
    psg_fast_axes = np.moveaxis(psg_fast_axes, 0, -1)
    psa_fast_axes = np.moveaxis(psa_fast_axes, 0, -1)

    # set up the drrp
    psg_pol = linear_polarizer(theta_pg, shape=[NMEAS])
    psg_wvp = linear_retarder(psg_fast_axes, psg_retardances, shape=[NPIX, NPIX, NMEAS])

    psa_wvp = linear_retarder(psa_fast_axes, psa_retardances, shape=[NPIX, NPIX, NMEAS])
    psa_pol = linear_polarizer(theta_pa, shape=[NMEAS])

    # Create power measurements
    power_measured = (psa_pol @ psa_wvp @ psg_wvp @ psg_pol)[..., 0, 0]

    return power_measured


def forward_simulate_pupil_avg(x, NMODES, NMEAS, mask=None):
    """Simulates a polarimetric measurement with
    spatially-varying data and averages over the pupil

    Parameters
    ----------
    x : ndarray or list
        parameter input vector
    NMODES: int
        Maximum zernike noll index to simulate

    Returns
    -------
    ndarray
        Power observed of shape NMEAS
    """
    power = forward_simulate(x, NMODES, NMEAS)
    if mask is None:
        power_pupil_avg = np.mean(power, axis=(0,1))
    else:
        power_pupil_avg = np.mean(power[mask==1], axis=0)

    return power_pupil_avg


def pack_ignorant_data(x, NMODES):
    # X is a reduced parameter vector, so we need to pack zeros everywhere else
    theta_pg = x[0] # Starting angle of the polarizer
    theta_pa = x[1] # Starting angle of the polarizer
    ret_psg = x[2]
    ang_psg = x[3]
    ret_psa = x[4]
    ang_psa = x[5]

    ret_psg_coeffs = np.zeros(NMODES)
    ang_psg_coeffs = np.zeros(NMODES)
    ret_psa_coeffs = np.zeros(NMODES)
    ang_psa_coeffs = np.zeros(NMODES)

    ret_psg_coeffs[0] = ret_psg
    ang_psg_coeffs[0] = ang_psg
    ret_psa_coeffs[0] = ret_psa
    ang_psa_coeffs[0] = ang_psa

    # Concatenate the input vars
    x0 = np.concatenate([np.array([theta_pg, theta_pa]),
                                   ret_psg_coeffs, ang_psg_coeffs,
                                   ret_psa_coeffs, ang_psa_coeffs])

    return x0

def forward_simulate_ignorant(x, NMODES, NMEAS, mask=None):

    # Stuff data with zeros
    x0 = pack_ignorant_data(x, NMODES)

    # do the pupil-averaged sim
    power = forward_simulate_pupil_avg(x0, NMODES, NMEAS, mask=mask)

    return power


if __name__ == "__main__":

    if len(sys.argv) > 1:
        NMODES = int(sys.argv[1])
        NMEAS = int(sys.argv[2])
        MAX_ITERS = int(sys.argv[3])
        BACKEND = sys.argv[4] # string, 'jax' or 'numpy'

    else:
        print("Using default arguments")
        NMODES = 128
        NMEAS = 24
        MAX_ITERS = 200
        BACKEND = "jax"

    assert BACKEND in ["jax", "numpy"]

    NPIX = 64
    N_PHOTONS = 1
    PLOT_INTERMEDIATES = False

    x, y = make_xy_grid(NPIX, diameter=2)
    r, t = cart_to_polar(x, y)

    A = circle(1, r)
    LS = circle(0.9, r)
    basis = construct_zern_basis(r, t, NMODES+1)

    ## GROUND TRUTH PARAMETERS
    SCALE = 100
    theta_pg = np.radians(0)
    theta_pa = np.radians(0)

    # Init retardance, PSG
    tnp.random.seed(24601)
    coeffs_spatial_ret_psg = tnp.random.random(len(basis)) / SCALE
    coeffs_spatial_ret_psg[0] = np.pi / 2

    # Init angle, PSG
    tnp.random.seed(8675309)
    coeffs_spatial_ang_psg = tnp.random.random(len(basis)) / (SCALE * 4)
    coeffs_spatial_ang_psg[0] = theta_pg #tnp.random.random()

    # Init retardance, PSA
    tnp.random.seed(24603)
    coeffs_spatial_ret_psa = tnp.random.random(len(basis)) / SCALE
    coeffs_spatial_ret_psa[0] = np.pi / 2

    # Init angle, PSA
    tnp.random.seed(24604)
    coeffs_spatial_ang_psa = tnp.random.random(len(basis)) / (SCALE * 4)
    coeffs_spatial_ang_psa[0] = theta_pa # tnp.random.random()

    x0_truth = np.concatenate([np.array([theta_pg, theta_pa]), coeffs_spatial_ret_psg, coeffs_spatial_ang_psg, coeffs_spatial_ret_psa, coeffs_spatial_ang_psa])
    power_experiment = forward_simulate(x0_truth, NMODES, NMEAS)

    # plot the ground truth retardance and angles
    retardance_psg = sum_of_2d_modes_wrapper(basis, coeffs_spatial_ret_psg)
    retardance_psa = sum_of_2d_modes_wrapper(basis, coeffs_spatial_ret_psa)
    angle_psg = sum_of_2d_modes_wrapper(basis, coeffs_spatial_ang_psg)
    angle_psa = sum_of_2d_modes_wrapper(basis, coeffs_spatial_ang_psa)

    if PLOT_INTERMEDIATES:
        plt.figure(figsize=[9.5,8])
        plt.subplot(221)
        vlim_ret = 3
        vlim_ang = .3
        plt.title("Polarization State Generator")
        plt.imshow(np.degrees(retardance_psg) / A, cmap="PuOr_r", vmin=90 - vlim_ret, vmax=90 + vlim_ret)
        plt.colorbar()
        plt.xticks([],[])
        plt.yticks([],[])
        plt.ylabel("Retardance")
        plt.subplot(222)
        plt.title("Polarization State Analyzer")
        plt.imshow(np.degrees(retardance_psa) / A, cmap="PuOr_r", vmin=90 - vlim_ret, vmax=90 + vlim_ret)
        plt.colorbar(label="Retardance, deg")
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(223)
        plt.imshow(np.degrees(angle_psg) / A, cmap="PiYG_r", vmin=np.mean(np.degrees(angle_psg[A==1])) - vlim_ang,
                                                            vmax=np.mean(np.degrees(angle_psg[A==1])) + vlim_ang)
        plt.colorbar()
        plt.xticks([],[])
        plt.yticks([],[])
        plt.ylabel("Fast Axis Angle")
        plt.subplot(224)
        plt.imshow(np.degrees(angle_psa) / A, cmap="PiYG_r", vmin=np.mean(np.degrees(angle_psa[A==1])) - vlim_ang,
                                                            vmax=np.mean(np.degrees(angle_psa[A==1])) + vlim_ang)
        plt.colorbar(label="Fast Axis Angle, deg")
        plt.xticks([],[])
        plt.yticks([],[])
        # plt.show()

    mean_power = []
    frame_photons = N_PHOTONS * power_experiment
    mean_power = np.mean(frame_photons[LS==1],axis=0)
    # for i in range(NMEAS):
    #     frame = power_experiment[..., i]
    #     mean_power.append(np.mean(((N_PHOTONS * frame[LS==1]))))

    plt.figure()
    plt.plot(mean_power, marker="o", linestyle="None", markersize=10)
    plt.ylabel("Mean Power, A.U")
    plt.xlabel("Frame Index")
    plt.title("Air Calibration")
    # plt.show()

    # Init some guess parameters
    tnp.random.seed(123)
    theta_pg = 0 #tnp.random.random()
    theta_pa = 0 #tnp.random.random()

    ret_pg = np.pi / 2 #+ tnp.random.random()
    ret_pa = np.pi / 2 #+ tnp.random.random()

    fast_pg = 0 #tnp.random.random()
    fast_pa = 0 #tnp.random.random()

    x0 = np.array([theta_pa, theta_pg, ret_pg, fast_pg, ret_pa, fast_pa])

    def loss(x, NMODES=NMODES, NMEAS=NMEAS):
        diff = forward_simulate_ignorant(x, NMODES, NMEAS, mask=LS) - mean_power
        diffsq = diff**2
        return np.sum(diffsq)

    def callback(xk):
        pbar.update(1)

    with tqdm(total=MAX_ITERS) as pbar:
        results = minimize(loss, x0=x0, callback=callback, method="L-BFGS-B", jac=False,
                        options={"maxiter": MAX_ITERS, "disp":False})

    psg_angles_highsample = np.linspace(0, 180, 1000)
    psg_angles = np.linspace(0, 180, NMEAS)
    x0_results = pack_ignorant_data(results.x, NMODES)
    power_modeled = forward_simulate_pupil_avg(x0_results, NMODES, NMEAS=1000, mask=LS)
    power_pupil = forward_simulate(x0_results, NMODES, NMEAS)

    if PLOT_INTERMEDIATES:
        plt.figure()
        plt.imshow(power_pupil[...,1] / A)
        plt.colorbar()

        plt.figure()
        plt.plot(psg_angles, mean_power, marker="o", linestyle="None", markersize=10, label="Power Measured")
        plt.plot(psg_angles_highsample, power_modeled, linestyle="solid", label="Power Modeled")
        plt.ylabel("Mean Power, A.U")
        plt.xlabel("PSG Angle, degrees")
        plt.title("Pupil-averaged Air Calibration")
        plt.legend()

    psg_pol = linear_polarizer(results.x[0], shape=[NPIX, NPIX, NMEAS])
    psa_pol = linear_polarizer(results.x[1], shape=[NPIX, NPIX, NMEAS])

    psg_wvp = linear_retarder(results.x[3] + np.radians(psg_angles), results.x[2], shape=[NPIX, NPIX, NMEAS])
    psa_wvp = linear_retarder(results.x[5] + (np.radians(psg_angles) * 2.5), results.x[4], shape=[NPIX, NPIX, NMEAS])

    PSG = (psg_wvp @ psg_pol)
    PSA = (psa_pol @ psa_wvp)
    Winv = drrp_data_reduction_matrix(PSG, PSA, invert=True)

    M_meas = Winv @ (N_PHOTONS * (power_experiment - 1e-7))[..., None]
    M_meas = np.reshape(M_meas[..., 0], [NPIX, NPIX, 4, 4])

    lyot_stop = circle(0.8, r)
    plt.style.use("default")

    if PLOT_INTERMEDIATES:
        plot_square(M_meas / M_meas[..., 0, 0, None, None] / lyot_stop[...,None,None], vmin=-1.1, vmax=1.1, common_cbar=False)


    # offset_retardation = 1e-1
    M_norm = M_meas / M_meas[..., 0, 0, None, None]
    # M_norm[..., 1, 1] = M_norm[..., 1, 1] - np.sin(offset_retardation)
    # M_norm[..., 2, 2] = M[..., 2, 2] + offset_retardation
    # M_norm[..., 3, 3] = M[..., 3, 3] + offset_retardation

    from katsu.mueller import retardance_from_mueller
    ret = retardance_from_mueller_taylor(M_norm * lyot_stop[..., None, None])
    ret -= np.mean(ret[lyot_stop==1])


    if PLOT_INTERMEDIATES:
        plt.figure(figsize=[12,4])
        plt.style.use("bmh")
        plt.subplot(121)
        plt.plot(psg_angles, mean_power, marker="o", linestyle="none", markersize=10, label="power measured")
        plt.plot(psg_angles_highsample, power_modeled, linestyle="solid", label="power modeled")
        plt.ylabel("mean power, a.u")
        plt.xlabel("psg angle, degrees")
        plt.title("pupil-averaged air calibration")
        plt.legend()

        plt.subplot(122)
        plt.title("retardance measured of air")
        plt.imshow(np.degrees(ret) / lyot_stop * lyot_stop, cmap="RdBu_r")
        plt.colorbar(label="retardance, deg")
        plt.xticks([],[])
        plt.yticks([],[])
        plt.show()

    ## performing the spatial calibration
    # init retardance, psg
    coeffs_spatial_ret_psg = np.zeros(len(basis))
    coeffs_spatial_ret_psg[0] = np.pi / 2

    # init angle, psg
    coeffs_spatial_ang_psg = (np.zeros(len(basis)))

    # Init retardance, PSA
    coeffs_spatial_ret_psa = np.zeros(len(basis))
    coeffs_spatial_ret_psa[0] = np.pi / 2

    # Init angle, PSA
    coeffs_spatial_ang_psa = np.zeros(len(basis))

    x0 = np.concatenate([np.array([theta_pg, theta_pa]), coeffs_spatial_ret_psg, coeffs_spatial_ang_psg, coeffs_spatial_ret_psa, coeffs_spatial_ang_psa])


    if BACKEND == "jax":
        set_backend_to_jax()

    power_experiment_masked = np.copy(power_experiment)
    if BACKEND == "jax":
        power_experiment_masked = power_experiment_masked.at[np.isnan(power_experiment)].set(1e-10)
    else:
        power_experiment_masked[np.isnan(power_experiment)] = 1e-10

    # need to define a new loss function
    from jax import value_and_grad

    def loss_jax(x, NMODES=NMODES, NMEAS=NMEAS):
        diff = forward_simulate(x, NMODES, NMEAS) - power_experiment_masked
        diffsq = diff**2
        return np.sum(diffsq[A==1])

    loss_fg = value_and_grad(loss_jax)

    print(f"Beginning timer for NMODES={NMODES}")
    t1 = time.perf_counter()
    with tqdm(total=MAX_ITERS) as pbar:

        if BACKEND == "jax":
            results_jax = minimize(loss_fg, x0=x0, callback=callback, method="L-BFGS-B", jac=True,
                            options={"maxiter": MAX_ITERS, "disp":False})
        if BACKEND == "numpy":
            results_numpy = minimize(loss_jax, x0=x0, callback=callback, method="L-BFGS-B", jac=False,
                            options={"maxiter": MAX_ITERS, "disp":False})
            results_jax = results_numpy
    runtime = time.perf_counter() - t1

    # Did we actually re-create the retardance / fast axis?
    vlim_ret = 3
    vlim_ang = 0.6
    cmap = "coolwarm"
    scale = 3600

    # extract the coefficients
    modeled_coeffs_spatial_ret_psg = results_jax.x[2 : 2 + 1*len(basis)]
    modeled_coeffs_spatial_ang_psg = results_jax.x[2 + 1*len(basis):(2 + 2*len(basis))]
    modeled_coeffs_spatial_ret_psa = results_jax.x[2 + 2*len(basis):(2 + 3*len(basis))]
    modeled_coeffs_spatial_ang_psa = results_jax.x[2 + 3*len(basis):(2 + 4*len(basis))]

    retardance_psg_modeled = sum_of_2d_modes_wrapper(basis, modeled_coeffs_spatial_ret_psg)
    retardance_psa_modeled = sum_of_2d_modes_wrapper(basis, modeled_coeffs_spatial_ret_psa)
    angle_psg_modeled = sum_of_2d_modes_wrapper(basis, modeled_coeffs_spatial_ang_psg)
    angle_psa_modeled = sum_of_2d_modes_wrapper(basis, modeled_coeffs_spatial_ang_psa)

    if PLOT_INTERMEDIATES:
        plt.figure(figsize=[9.5,8])
        plt.subplot(221)
        plt.title("Polarization State Generator")
        plt.imshow(np.degrees(retardance_psg_modeled - retardance_psg) * scale / A, cmap=cmap, vmin=-vlim_ret, vmax=vlim_ret)
        plt.colorbar()
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(222)
        plt.title("Polarization State Analyzer")
        plt.imshow(np.degrees(retardance_psa_modeled - retardance_psa) * scale / A, cmap=cmap, vmin=-vlim_ret, vmax=vlim_ret)
        plt.colorbar(label="Retardance Residuals, arcsec")
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(223)
        plt.imshow(np.degrees(angle_psg_modeled - angle_psg) * scale / A, cmap=cmap, vmin=-vlim_ang, vmax=vlim_ang)
        plt.colorbar()
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(224)
        plt.imshow(np.degrees(angle_psa_modeled - angle_psa) * scale / A, cmap=cmap, vmin=-vlim_ang, vmax=vlim_ang)
        plt.colorbar(label="Fast Axis Angle Residuals, arcsec")
        plt.xticks([],[])
        plt.yticks([],[])


    ## PERFORM THE POLARIMETRIC DATA REDUCTION
    psg_pol = linear_polarizer(results_jax.x[0], shape=[NPIX, NPIX, NMEAS])
    psa_pol = linear_polarizer(results_jax.x[1], shape=[NPIX, NPIX, NMEAS])

    psg_retardances, psa_retardances = [], []
    psg_fast_axes, psa_fast_axes = [], []
    psg_angles_radians = np.radians(psg_angles)

    for rot_psg in psg_angles_radians:

        rot_psa = rot_psg * 2.5

        # Get the rotated spatial basis
        basis_psg = np.asarray(construct_zern_basis(r, t + rot_psg, NMODES+1))
        basis_psa = np.asarray(construct_zern_basis(r, t + rot_psa, NMODES+1))

        # compute retardances
        psg_retardance = sum_of_2d_modes_wrapper(basis_psg, modeled_coeffs_spatial_ret_psg)
        psa_retardance = sum_of_2d_modes_wrapper(basis_psa, modeled_coeffs_spatial_ret_psa)

        # compute fast axes
        psg_fast_axis = sum_of_2d_modes_wrapper(basis_psg, modeled_coeffs_spatial_ang_psg)
        psa_fast_axis = sum_of_2d_modes_wrapper(basis_psa, modeled_coeffs_spatial_ang_psa)

        # store arrays in list
        psg_retardances.append(psg_retardance)
        psa_retardances.append(psa_retardance)
        psg_fast_axes.append(psg_fast_axis + rot_psg)
        psa_fast_axes.append(psa_fast_axis + rot_psa)

    psg_retardances = np.asarray(psg_retardances)
    psa_retardances = np.asarray(psa_retardances)
    psg_fast_axes = np.asarray(psg_fast_axes)
    psa_fast_axes = np.asarray(psa_fast_axes)

    psg_retardances = np.moveaxis(psg_retardances, 0, -1)
    psa_retardances = np.moveaxis(psa_retardances, 0, -1)
    psg_fast_axes = np.moveaxis(psg_fast_axes, 0, -1)
    psa_fast_axes = np.moveaxis(psa_fast_axes, 0, -1)

    psg_wvp = linear_retarder(psg_fast_axes, psg_retardances, shape=[NPIX, NPIX, NMEAS])
    psa_wvp = linear_retarder(psa_fast_axes, psa_retardances, shape=[NPIX, NPIX, NMEAS])

    PSG = (psg_wvp @ psg_pol)
    PSA = (psa_pol @ psa_wvp)
    Winv = drrp_data_reduction_matrix(PSG, PSA, invert=True)

    M_meas = Winv @ (N_PHOTONS * power_experiment_masked)[..., None]
    M_meas = np.reshape(M_meas[..., 0], [NPIX, NPIX, 4, 4])

    # offset_retardation = 1e-1
    M_norm = M_meas / M_meas[..., 0, 0, None, None]
    # M_norm = M_norm.at[..., 1, 1].set(M_norm[..., 1, 1] - np.sin(offset_retardation))

    plt.style.use("default")
    if PLOT_INTERMEDIATES:
        plot_square(M_meas / M_meas[...,0,0, None, None] / A[..., None, None], vmin=None, vmax=None, common_cbar=False)



    from katsu.mueller import retardance_from_mueller
    from katsu.mueller import decompose_retarder
    M_ret = decompose_retarder(M_norm)
    ret = retardance_from_mueller_taylor(M_ret * lyot_stop[..., None, None])
    ret -= np.mean(ret[lyot_stop==1])

    if PLOT_INTERMEDIATES:
        plt.figure(figsize=[12,4])
        plt.style.use("bmh")
        plt.subplot(121)
        plt.plot(psg_angles, mean_power, marker="o", linestyle="None", markersize=10, label="power measured")
        plt.plot(psg_angles_highsample, power_modeled, linestyle="solid", label="power modeled")
        plt.ylabel("mean power, a.u")
        plt.xlabel("psg angle, degrees")
        plt.title("pupil-averaged air calibration")
        plt.legend()

        plt.subplot(122)
        plt.title("retardance measured of air")
        plt.imshow(np.degrees(ret) * lyot_stop / lyot_stop, cmap="RdBu_r")
        plt.colorbar(label="retardance, deg")
        plt.xticks([],[])
        plt.yticks([],[])


        plt.show()

    print(f"Finished in {runtime} seconds for NMODES={NMODES}")
