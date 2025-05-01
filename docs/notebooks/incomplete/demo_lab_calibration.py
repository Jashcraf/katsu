import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
from tqdm import tqdm
import numpy as tnp
import time
import sys
from pathlib import Path
import ipdb
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import warnings


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
sys.path.append("C:/Users/Work/Desktop/derp_control/")
from derpy.writing import read_experiment
from derpy.experiments import forward_calibrate, forward_simulate

# Data loading
DATA_PTH = Path.home() / "Box/97_Data/derp/20250210_GPI"
CALIBRATION_ID = DATA_PTH / "GPI_HWP_center_air_calibration.msgpack"
EXPERIMENT_ID = DATA_PTH / "GPI_HWP_center_measure_0.msgpack"

# TODO: Make these not redundant
WVL_ID = 4
WAVELENGTH_SELECT = 1500  # nm

PLOT_INTERMEDIATES = True
MASK_RAD = 0.5 # from 0 to 1, 1 being the full circle
MODE = "both" # "left", "right", "both"
BAD_FRAMES_CAL = [] # [3, 13 -1], _, [8]
BAD_FRAMES = [4]  # [-9], _, [-11]
PLOT_IMAGES = False
N_PHOTONS = 1
# --------------------------------


# Plotting stuff
plt.style.use('bmh')
okabe_colorblind8 = ['#E69F00','#56B4E9','#009E73',
                     '#F0E442','#0072B2','#D55E00',
                     '#CC79A7','#000000']

plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=okabe_colorblind8)


"""Data Loading and Preprocessing methods
"""
def calibrate_experiment(x, experiment):

    experiment.psg_pol_angle = x[0]
    experiment.psg_starting_angle = np.degrees(x[1])
    experiment.psg_wvp_ret = x[2]
    experiment.psa_pol_angle = x[3]
    experiment.psa_starting_angle = np.degrees(x[4])
    experiment.psa_wvp_ret = x[5]

    return experiment

def load_and_pre_process_data(pth, wavelength_id, bad_frames,
                              reference_frame_index=0, reference_channel="left",
                              shiftx=0, shifty=0, mask_rad=0.7, crop=25):
    """Load data from derpy.Experiment class, mask bad frames, and center the images

    Parameters
    ----------
    pth : str or PosixPath
        path to the derpy.Experiment class containing the appropriate data
    wavelength_id : int
        index of Experiment.images corresponding to the wavelength of interest
    bad_frames : list of int
        list of indices that contain frames to be ignored
    reference_frame_index : int, optional
        which frame to use as the centering reference, by default 0
    reference_channel : str, optional
        which channel to use as the reference frame for the DRRP data, by default "left"

    Returns
    -------
    ndarray
        Array containing images of shape frame, channel, row, col
    """

    assert reference_channel in ["left", "right"]

    if reference_channel == "left":
        reference_channel = 0
        other_channel = 1
    elif reference_channel == "right":
        reference_channel = 1
        other_channel = 0

    experiment = read_experiment(pth)

    # Create a mask to use as a centering reference
    # x = np.linspace(-1, 1, experiment.images.shape[2])
    # x, y = np.meshgrid(x, x)
    # r = np.sqrt(x**2 + y**2)
    # mask = np.zeros_like(experiment.images[0, 0, 0])
    # ipdb.set_trace()
    # extract data
    # experiment.images is shape wavelength, frame, channel, row, col
    # Start by masking out bad frames
    images = experiment.images[wavelength_id]
    powers = experiment.mean_power_left[wavelength_id]
    masked_images = []
    mean_power = []
    psg_angles = []
    for i, img in enumerate(images):
        if i not in bad_frames:
            masked_images.append(img)
            mean_power.append(powers[i])
            psg_angles.append(experiment.psg_positions_relative[i])

    masked_images = np.asarray(masked_images)
    mean_power = np.asarray(mean_power)

    # now do the phase cross-correlation to center the images
    reference_image = masked_images[reference_frame_index, reference_channel]
    reference_image = shift(reference_image, (shiftx, shifty))
    cropped_reference = reference_image[crop:-crop, crop:-crop]

    # perform a cropping
    # This just does the registration on the other channel in the first frame
    dpix, error, diffphase = phase_cross_correlation(reference_image, 
                                                      masked_images[reference_frame_index][other_channel])
    shifted_other = shift(masked_images[reference_frame_index][other_channel], dpix)
    cropped_other = shifted_other[crop:-crop, crop:-crop]

    # register the right frame to the reference image
    for i, img in enumerate(masked_images):
        if i != reference_frame_index:

            # phase cross correlation to the left frame
            dpix, error, diffphase = phase_cross_correlation(reference_image, img[0])
            masked_images[i, 0] = shift(img[0], dpix)
            
            # phase cross correlation to the right frame
            dpix, error, diffphase = phase_cross_correlation(reference_image, img[1])
            masked_images[i, 1] = shift(img[1], dpix)

    # Scale down to a relative power of 0.5
    scale_power = 2 * np.max(experiment.mean_power_left[WVL_ID])
    masked_images /= scale_power

    cropped_images = np.zeros([*masked_images.shape[:2],
                              masked_images.shape[2] - 2 * crop,
                              masked_images.shape[3] - 2 * crop])

    # crop all of the images
    for i in range(masked_images.shape[0]):
        if i == reference_frame_index:
            cropped_images[i, 0] = cropped_reference / scale_power
            cropped_images[i, 1] = cropped_other / scale_power
        else:
            cropped_images[i, 0] = masked_images[i][0,crop:-crop, crop:-crop]
            cropped_images[i, 1] = masked_images[i][1,crop:-crop, crop:-crop]


    return cropped_images, mean_power, psg_angles


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
        DO_FINITE_DIFF = sys.argv[5] # bool, True or False
    else:
        print("Using default arguments")
        NMODES = 32
        NMEAS = 50 
        MAX_ITERS = 100
        BACKEND = "jax"
        DO_FINITE_DIFF = False
       
    assert BACKEND in ["jax", "numpy"]

    frames, mean_power, psg_angles = load_and_pre_process_data(EXPERIMENT_ID,
                                                               WVL_ID, BAD_FRAMES,
                                                               shiftx=13, shifty=-15)
    NMEAS -= len(BAD_FRAMES)
    exp = read_experiment(EXPERIMENT_ID)
    NPIX = frames.shape[-1]
    power_experiment = frames

    x, y = make_xy_grid(NPIX, diameter=2)
    r, t = cart_to_polar(x, y)

    A = circle(1, r)
    LS = circle(0.9, r)
    basis = construct_zern_basis(r, t, NMODES+1)


    plt.figure()
    plt.plot(mean_power, marker="o", markersize=10, label="left")
    plt.ylabel("Mean Power, A.U")
    plt.xlabel("Frame Index")
    plt.title("Air Calibration")
    plt.show()

    # Init some guess parameters
    tnp.random.seed(123)
    theta_pg = 0 #tnp.random.random()
    theta_pa = 0 #tnp.random.random()

    ret_pg = np.pi / 2 #+ tnp.random.random()
    ret_pa = np.pi / 2 #+ tnp.random.random()

    fast_pg = 0 #tnp.random.random()
    fast_pa = 0 #tnp.random.random()

    x0 = np.array([theta_pa, theta_pg, ret_pg, fast_pg, ret_pa, fast_pa])

    lyot_stop = circle(0.8, r)
    psg_angles_highsample = np.linspace(0, 180, 1000)

    if DO_FINITE_DIFF:
        
        def loss(x, NMODES=NMODES, NMEAS=NMEAS):
            scaled_mean_power_left = mean_power / mean_power.max() / 2
            diff = forward_simulate_ignorant(x, NMODES, NMEAS, mask=LS) - scaled_mean_power_left
            diffsq = diff**2
            return np.sum(diffsq)


        results = minimize(loss, x0=x0, method="L-BFGS-B", jac=False,
                        options={"maxiter": 10, "disp":False})



        psg_angles = np.linspace(0, 180, NMEAS)
        x0_results = pack_ignorant_data(results.x, NMODES)
        power_modeled = forward_simulate_pupil_avg(x0_results, NMODES, NMEAS=1000, mask=LS)
        power_pupil = forward_simulate(x0_results, NMODES, NMEAS)

        plt.figure()
        plt.imshow(power_pupil[...,1] / A)
        plt.colorbar()

        plt.figure()
        plt.plot(psg_angles, mean_power / mean_power.max() / 2, marker="o", linestyle="None", markersize=10, label="Power Measured")
        plt.plot(psg_angles_highsample, power_modeled, linestyle="solid", label="Power Modeled")
        plt.ylabel("Mean Power, A.U")
        plt.xlabel("PSG Angle, degrees")
        plt.title("Pupil-averaged Air Calibration")
        plt.legend()
        plt.show()

        psg_pol = linear_polarizer(results.x[0], shape=[NPIX, NPIX, NMEAS])
        psa_pol = linear_polarizer(results.x[1], shape=[NPIX, NPIX, NMEAS])

        psg_wvp = linear_retarder(results.x[3] + np.radians(psg_angles), results.x[2], shape=[NPIX, NPIX, NMEAS])
        psa_wvp = linear_retarder(results.x[5] + (np.radians(psg_angles) * 2.5), results.x[4], shape=[NPIX, NPIX, NMEAS])

        PSG = (psg_wvp @ psg_pol)
        PSA = (psa_pol @ psa_wvp)
        Winv = drrp_data_reduction_matrix(PSG, PSA, invert=True)

        # make frames the right shape
        frames = frames[:, 0]
        frames = np.moveaxis(frames, 0, -1)

        M_meas = Winv @ (N_PHOTONS * (frames - 1e-7))[..., None]
        M_meas = np.reshape(M_meas[..., 0], [NPIX, NPIX, 4, 4])

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
    
    psg_angles = np.asarray(psg_angles)
    power_experiment = power_experiment[:, 0]
    power_experiment = np.moveaxis(power_experiment, 0, -1)
    power_experiment_masked = np.copy(power_experiment)

    if BACKEND == "jax":
        power_experiment_masked = power_experiment_masked.at[np.isnan(power_experiment)].set(1e-10)
    else:
        power_experiment_masked[np.isnan(power_experiment)] = 1e-10

    # need to define a new loss function
    from jax import value_and_grad

    def loss_jax(x, NMODES=NMODES, NMEAS=NMEAS):
        diff = forward_simulate(x, NMODES, NMEAS) - power_experiment_masked
        diffsq = diff[A==1]**2
        return np.sum(diffsq)

    loss_fg = value_and_grad(loss_jax)

    print(f"Beginning timer for NMODES={NMODES}")
    t1 = time.perf_counter()
    with tqdm(total=MAX_ITERS) as pbar:

        if BACKEND == "jax":
            results_jax = minimize(loss_fg, x0=x0, method="L-BFGS-B", jac=True,
                            options={"maxiter": MAX_ITERS, "disp":True, "ftol":1e-10})
        if BACKEND == "numpy":
            results_numpy = minimize(loss_jax, x0=x0, method="L-BFGS-B", jac=False,
                            options={"maxiter": MAX_ITERS, "disp":False})
            results_jax = results_numpy
    runtime = time.perf_counter() - t1

    # Did we actually re-create the retardance / fast axis?
    vlim_ret = 180
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
        plt.imshow(np.degrees(retardance_psg_modeled) * scale / A, cmap=cmap, vmin=-vlim_ret, vmax=vlim_ret)
        plt.colorbar()
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(222)
        plt.title("Polarization State Analyzer")
        plt.imshow(np.degrees(retardance_psa_modeled) * scale / A, cmap=cmap, vmin=-vlim_ret, vmax=vlim_ret)
        plt.colorbar(label="Retardance Residuals, arcsec")
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(223)
        plt.imshow(np.degrees(angle_psg_modeled) * scale / A, cmap=cmap, vmin=-vlim_ang, vmax=vlim_ang)
        plt.colorbar()
        plt.xticks([],[])
        plt.yticks([],[])
        plt.subplot(224)
        plt.imshow(np.degrees(angle_psa_modeled) * scale / A, cmap=cmap, vmin=-vlim_ang, vmax=vlim_ang)
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

    ipdb.set_trace()

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
