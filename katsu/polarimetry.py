from .mueller import linear_retarder, linear_polarizer, linear_diattenuator, _empty_mueller, decompose_retarder
from .katsu_math import broadcast_kron, broadcast_outer, condition_number, RMS_calculator, propagated_error, np
from scipy.optimize import curve_fit



def full_mueller_polarimetry(thetas, power, angular_increment,
                             return_condition_number=False,
                             Min=None,
                             starting_angles={'psg_polarizer': 0,
                                              'psg_waveplate': 0,
                                              'psa_waveplate': 0,
                                              'psa_polarizer': 0},
                             starting_polarization={'psg_Tmin': 0,
                                                    'psg_ret': np.pi / 2,
                                                    'psa_Tmin': 0,
                                                    'psa_ret': np.pi / 2},
                             starting_anglestep={'psg_step': 1,
                                                 'psa_step': 1}):
    """conduct a full mueller polarimeter measurement from a series of power
    measurements

    Parameters
    ----------
    thetas : numpy.ndarray
        np.linspace(starting_angle,ending_angle,number_of_measurements)
    power : numpy.ndarray
        3D array of power recorded from the polarimeter. The first two
        dimensions are spatial, and the last is temporal i.e. power[...,0] is
        the first measurement, power[...,0] is the second measurement
    angular_increment : float
        The fractional angular increment that the PSA rotates compared to
        the PSG. This can be computed by the ratio
        PSA_increment / PSG_increment.
    return_condition_number : bool, optional
        returns condition number of the data reduction matrix. by default False
    Min : numpy.ndarray
        if provided, is the "true" Mueller matrix. This allows us to
        simulate full mueller polarimetry. by default None
    starting_angles : dict
        the starting angles (in radians) of the optics that make up the
        polarization state generator and analyzer.

        Keys are:
        -------------------------------------------
        psg_polarizer = polarization state generator polarizer angle
        psg_waveplate = polarization state generator quarter wave plate angle
        psa_waveplate = polarization state analyzer polarizer angle
        psa_qwp = polarization state analyzer quarter wave plate angle


    Returns
    -------
    numpy.ndarray
        Mueller matrix measured by the polarimeter
    """

    nmeas = len(thetas)
    psg_angles = thetas * starting_anglestep['psg_step']
    psa_angles = thetas * starting_anglestep['psa_step'] * angular_increment

    # handle the case of imaging v.s. single-pixel polarimetry
    if isinstance(power, np.ndarray):
        if power.ndim > 1:
            frame_shape = power.shape[:-1]

        else:
            frame_shape = ()

    psg_tmin = starting_polarization['psg_Tmin']
    psg_ret = starting_polarization['psg_ret']
    psg_theta = starting_angles['psg_waveplate'] + psg_angles

    psa_tmin = starting_polarization['psa_Tmin']
    psa_ret = starting_polarization['psa_ret']
    psa_theta = starting_angles['psa_waveplate'] + psa_angles

    psg_qwp = linear_retarder(psg_theta, psg_ret, shape=[*frame_shape, nmeas])
    psg_hpl = linear_diattenuator(starting_angles['psg_polarizer'],
                                  Tmin=psg_tmin, shape=[*frame_shape, nmeas])

    psa_qwp = linear_retarder(psa_theta, psa_ret, shape=[*frame_shape, nmeas])
    psa_hpl = linear_diattenuator(starting_angles['psa_polarizer'],
                                  Tmin=psa_tmin, shape=[*frame_shape, nmeas])
    Mg = psg_qwp @ psg_hpl
    Ma = psa_hpl @ psa_qwp

    PSA = Ma[..., 0, :]
    PSG = Mg[..., :, 0]

    # polarimetric data reduction matrix, flatten Mueller matrix dimension
    Wmat = broadcast_kron(PSA[..., np.newaxis], PSG[..., np.newaxis])
    Wmat = Wmat.reshape([*Wmat.shape[:-2], 16])
    Winv = np.linalg.pinv(Wmat)
    power_expand = power[..., np.newaxis]

    # Do the data reduction
    M_meas = Winv @ power_expand
    M_meas = M_meas[..., 0]

    return M_meas.reshape([*M_meas.shape[:-1], 4, 4])


def stokes_sinusoid(theta, a0, b2, a4, b4):
    """sinusoidal response of a single rotating retarder full stokes
    polarimeter.

    Parameters
    ----------
    theta : float
        angle of QWP
    a0 : float
        zero frequency coefficient
    b2 : float
        sin(2\theta) coefficient
    a4 : float
        cos(4\theta) coefficient
    b4 : float
        sin(4\theta) coefficient

    Returns
    -------
    numpy.ndarray
        sinusoidal response of the SRRP
    """
    return a0 + b2*np.sin(2*theta) + a4*np.cos(4*theta) + b4*np.sin(4*theta)


def full_stokes_polarimetry(thetas, Sin=None, power=None, return_coeffs=False):
    """conduct a full stokes polarimeter measurement

    Parameters
    ----------
    thetas : numpy.ndarray
        rotation angles of the QWP fast axis w.r.t. the horizontal polarizer
    Sin : numpy.ndarray, optional
        input stokes vector, used for simulating polarimetry. by default None
    power : numpy.ndarray, optional
        powers measured on detector for each angle theta, by default None
    return_coeffs : bool, optional
        option to return the stokes sinusoid coefficients. Useful for
        evaluating curve fit quality. by default None

    Returns
    -------
    numpy.ndarray
        array containing the Stokes vector measured. Also returns coefficients
        of curve fit of return_coeffs==True.s
    """

    nmeas = len(thetas)
    Pmat = np.zeros([nmeas])

    # Retarder needs to rotate 2pi, break up by nmeas
    th = thetas

    for i in range(nmeas):

        # Mueller Matrix of analyzer
        M = linear_polarizer(0) @ linear_retarder(th[i], np.pi/2)

        # The top row is the analyzer vector
        analyzer = M[0, :]

        # Record the power
        if power is not None:
            Pmat[i] = power[i]
        else:
            Pmat[i] = np.dot(analyzer, Sin)

    popt, pcov = curve_fit(stokes_sinusoid,
                           th,
                           Pmat,
                           p0=(1, 1, 1, 1))

    a0 = popt[0]
    b2 = popt[1]
    a4 = popt[2]
    b4 = popt[3]

    # Compute the Stokes Vector
    S0 = 2*(a0 - a4)
    S1 = 4 * a4
    S2 = 4 * b4
    S3 = -2 * b2

    if return_coeffs:
        return np.array([S0, S1, S2, S3]), popt
    else:
        return np.array([S0, S1, S2, S3])


def _full_mueller_polarimetry(thetas,power=1,return_condition_number=False,Min=None,
                             starting_angles={'psg_polarizer':0,
                                              'psg_qwp':0,
                                              'psa_qwp':0,
                                              'psa_polarizer':0}):
    """DEPRECIATED: Replaced with a broadcasted variant
    conduct a full mueller polarimeter measurement

    Parameters
    ----------
    thetas : numpy.ndarray
        np.linspace(starting_angle,ending_angle,number_of_measurements)
    power : float, optional
        power recorded on a given pixel. Defaults to 1
    return_condition_number : bool, optional
        returns condition number of the data reduction matrix. by default False
    Min : numpy.ndarray
        if provided, is the "true" Mueller matrix. This allows us to
        simulate full mueller polarimetry. by default None
    starting_angles : dict
        the starting angles (in radians) of the optics that make up the polarization state generator and analyzer. 
        Keys are:
        -------------------------------------------
        psg_polarizer = polarization state generator polarizer angle
        psg_qwp = polarization state generator quarter wave plate angle
        psa_polarizer = polarization state analyzer polarizer angle
        psa_qwp = polarization state analyzer quarter wave plate angle


    Returns
    -------
    numpy.ndarray
        Mueller matrix measured by the polarimeter
    """
    nmeas = len(thetas)

    Wmat = np.zeros([nmeas,16])
    Pmat = np.zeros([nmeas])
    th = thetas

    for i in range(nmeas):

        # Mueller Matrix of Generator using a QWR
        Mg = linear_retarder(starting_angles['psg_qwp']+th[i],np.pi/2) @ linear_polarizer(starting_angles['psg_polarizer'])

        # Mueller Matrix of Analyzer using a QWR
        Ma = linear_polarizer(starting_angles['psa_polarizer']) @ linear_retarder(starting_angles['psa_qwp']+th[i]*5,np.pi/2)

        ## Mueller Matrix of System and Generator
        # The Data Reduction Matrix
        Wmat[i,:] = np.kron(Ma[0,:],Mg[:,0])

        # A detector measures the first row of the analyzer matrix and first column of the generator matrix
        if Min is not None:
            Pmat[i] = (Ma[0,:] @ Min @ Mg[:,0]) * power
        else:
            Pmat[i] = power[i]

    # Compute Mueller Matrix with Moore-Penrose Pseudo Inverse
    # Calculation appears to be sensitive to the method used to compute the inverse! There's something I guess
    M = np.linalg.pinv(Wmat) @ Pmat
    M = np.reshape(M,[4,4])

    if return_condition_number == True:
        return M,condition_number(Wmat)

    else:
        return M
    
# Function that makes the Mueller matrix using the calibration parameters a1, w1, w2, r2, and r2. Set these to 0 for an uncalibrated matrix
def q_calibrated_full_mueller_polarimetry(thetas, a1, w1, w2, r1, r2, I_vert, I_hor, M_in=None):
    """Full Mueller polarimetry using measurements of Q and calibration parameters. 
    Gives a calibrated Mueller matrix with the parameters, or set a1, w1, w2, r1, and r2 to zero for an uncalibrated matrix.
    Parameters
    ----------
    thetas : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    I_hor : array
        measured intensity of the horizontal polarization spot from the Wollaston prism
    I_vert : array
        measured intensity of the vertical polarization spot from the Wollaston prism
    Returns
    -------
    M : array
        4x4 Mueller matrix for the measured sample. """
    nmeas = len(thetas)  # Number of measurements
    Wmat1 = np.zeros([nmeas, 16])
    Pmat1 = np.zeros([nmeas])
    Wmat2 = np.zeros([nmeas, 16])
    Pmat2 = np.zeros([nmeas])
    th = thetas
    unnormalized_Q = I_hor - I_vert   # Difference in intensities measured by the detector
    unnormalized_I_total = I_vert + I_hor
    Q = unnormalized_Q/np.max(unnormalized_I_total)
    I_total = unnormalized_I_total/np.max(unnormalized_I_total)
    # Both Q and I should be normalized by the total INPUT flux, but we don't know this value. The closest we can guess is the maximum of the measured intensity
    # This assumes the input flux is constant over time. Could be improved with a beam splitter that lets us monitor the input flux over time

    for i in range(nmeas):
        # Mueller Matrix of generator (linear polarizer and a quarter wave plate)
        Mg = linear_retarder(th[i]+w1, np.pi/2+r1) @ linear_polarizer(0+a1)

        # Mueller Matrix of analyzer (one channel of the Wollaston prism is treated as a linear polarizer)
        Ma = linear_retarder(th[i]*5+w2, np.pi/2+r2)

        # Data reduction matrix. Taking the 0 index ensures that intensity is the output
        Wmat1[i,:] = np.kron((Ma)[0,:], Mg[:,0]) # for the top row, using intensities
        Wmat2[i,:] = np.kron((Ma)[1,:], Mg[:,0]) # for the bottom 3 rows, using Q

        # M_in is some example Mueller matrix. Providing this input will test theoretical Mueller matrix. Otherwise, the raw data is used
        if M_in is not None:
            Pmat1[i] = (Ma[0,:] @ M_in @ Mg[:,0])
            Pmat2[i] = (Ma[1,:] @ M_in @ Mg[:,0])
        else:
            Pmat1[i] = I_total[i]  #Pmat is a vector of measurements (either I or Q)
            Pmat2[i] = Q[i] 

    # Compute Mueller matrix using Moore-Penrose pseudo invervse
    M1 = np.linalg.pinv(Wmat1) @ Pmat1
    M1 = np.reshape(M1, [4,4])

    M2 = np.linalg.pinv(Wmat2) @ Pmat2
    M2 = np.reshape(M2, [4,4])

    M = np.zeros([4,4])
    M[0,:] = M1[0,:]
    M[1:4,:] = M2[1:4,:]

    return M

# Define the identity matrix and other matrices which are useful for the Mueller calculus
M_identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
A = np.array([1, 0, 0, 0])
B = np.array([[1], [0], [0], [0]])
C = np.array([0, 1, 0, 0])

# # The next function should work equivalently
# def q_calibration_function(t, a1, w1, w2, r1, r2):
#     """Function that models the Mueller calculus for the DRRP system and is used to calculate the calibration parameters.
#     t : array
#         angles of the first quarter wave plate
#     a1 : float
#         calibration parameter for the offset angle of the first linear polarizer
#     w1 : float
#         calibration parameter for the offset angle of the first quarter-wave plate fast axis.
#     w2 : float
#         calibration parameter for the offset angle of the second quarter-wave plate fast axis.
#     r1 : float
#         calibration parameter for the retardance offset of the first quarter-wave plate. 
#     r2 : float
#         calibration parameter for the retardance offset of the second quarter-wave plate.
#     Returns:
#         An array of predictions for measured Q values."""
#     prediction = [None]*len(t)
#     for i in range(len(t)):
#         prediction[i] = float(C @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M_identity @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
#     return prediction


def q_output_simulation_function(t, a1, w1, w2, r1, r2, M_in=None):
    """Function that models the Mueller calculus for the DRRP system and is used to calculate the calibration parameters.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values."""
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(C @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Function that is useful for generating intensity values for a given sample matrix and offset parameters
def I_output_simulation_function(t, a1, w1, w2, r1, r2, M_in=None):
    """Function to generate TOTAL intensity values measured with a given Mueller matrix and offset parameters.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values."""
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(A  @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# Basically the same as above, but with an optional input matrix to simulate data
def single_output_simulation_function(t, a1, a2, w1, w2, r1, r2, LPA_angle=0, M_in=None):
    """Function to generate intensity values for one polarization at a time. Default is horizontal, with LPA=0. For vertical, set LPA=pi/2.
    Parameters
    ----------
    t : array
        angles of the first quarter wave plate
    a1 : float
        calibration parameter for the offset angle of the first linear polarizer
    a2 : float
        calibration parameter for the offset angle of the second linear polarizer (could be just one channel of the Wollaston prism)
    w1 : float
        calibration parameter for the offset angle of the first quarter-wave plate fast axis.
    w2 : float
        calibration parameter for the offset angle of the second quarter-wave plate fast axis.
    r1 : float
        calibration parameter for the retardance offset of the first quarter-wave plate. 
    r2 : float
        calibration parameter for the retardance offset of the second quarter-wave plate.
    LPA_angle : float
        angle of the analyzing linear polarizer. Default is 0 for horizontal. Set to pi/2 for vertical.
    M_in : array
        optional 4x4 Mueller matrix to simulate data. By default None, which uses the identity matrix for air. 
    Returns
    -------
    prediction : array
        An array of predictions for measured Q values.    """
    if M_in is None:
        M = M_identity
    else:
        M = M_in

    prediction = [None]*len(t)
    for i in range(len(t)):
        prediction[i] = float(A @ linear_polarizer(LPA_angle+a2) @ linear_retarder(5*t[i]+w2, np.pi/2+r2) @ M @ linear_retarder(t[i]+w1, np.pi/2+r1) @ linear_polarizer(a1) @ B)
    return prediction


# The function that gives everything you want to know at once
def q_ultimate_polarimetry(cal_angles, cal_vert_intensity, cal_hor_intensity, sample_angles, sample_vert_intensity, sample_hor_intensity):
    """Function that calculates the Mueller matrix of a sample and other relevant information.
    cal_angles and sample_angles could be the same, or could be different.
    Parameters
    ----------
    cal_angles : array
        angles of the first quarter wave plate for calibration
    cal_vert_intensity : array
        measured intensity of the vertical polarization spot from the Wollaston prism for calibration
    cal_hor_intensity : array
        measured intensity of the horizontal polarization spot from the Wollaston prism for calibration
    sample_angles : array
        angles of the first quarter wave plate when taking data with the sample
    sample_vert_intensity : array
        measured intensity of the vertical polarization spot from the Wollaston prism when taking data with the sample
    sample_hor_intensity : array
        measured intensity of the horizontal polarization spot from the Wollaston prism when taking data with the sample
    Returns
    -------
    M_Sample : array
        4x4 Mueller matrix for the sample
    retardance : float
        extracted retardance of the sample in waves
    M_Cal : array
        4x4 Mueller matrix for the calibration (should resemble the identity matrix)
    RMS_Error : float
        root mean square error of the calibration matrix
    Retardance_Error : float
        error of the retardance value, assuming the RMS error from the calibration matrix is the same for all elements of the sample matrix.
    """
    ICal = cal_hor_intensity + cal_vert_intensity
    QCal = cal_hor_intensity - cal_vert_intensity 
    initial_guess = [0, 0, 0, 0, 0]
    parameter_bounds = ([-np.pi, -np.pi, -np.pi, -np.pi/2, -np.pi/2], [np.pi, np.pi, np.pi, np.pi/2, np.pi/2])

    # Find parameters from calibration 
    normalized_QCal = QCal/(max(ICal)) # This should be normalized by the input intensity, but we don't know that so use the max of the measured intensity instead as an approximation
    # popt, pcov = curve_fit(q_calibration_function, cal_angles, normalized_QCal, p0=initial_guess, bounds=parameter_bounds)
    popt, pcov = curve_fit(q_output_simulation_function, cal_angles, normalized_QCal, p0=initial_guess, bounds=parameter_bounds)
    print(popt, "Fit parameters for a1, w1, w2, r1, and r2. 1 for generator, 2 for analyzer")

    # The calibration matrix (should be close to identity) to see how well the parameters compensate
    MCal = q_calibrated_full_mueller_polarimetry(cal_angles, popt[0], popt[1], popt[2], popt[3], popt[4], cal_vert_intensity, cal_hor_intensity)
    MCal = MCal/np.max(np.abs(MCal))
    RMS_Error = RMS_calculator(MCal)

    # Use the parameters found above from curve fitting to construct the actual Mueller matrix of the sample
    MSample = q_calibrated_full_mueller_polarimetry(sample_angles, popt[0], popt[1], popt[2], popt[3], popt[4], sample_vert_intensity, sample_hor_intensity)
    MSample = MSample/np.max(np.abs(MSample))

    np.set_printoptions(suppress=True) # Suppresses scientific notation, keeps decimal format

    # Use the polar decomposition of the retarder matrix 
    r_decomposed_MSample = decompose_retarder(MSample, normalize=True)
    # retardance = np.arccos(np.trace(normalized_decompose_retarder(r_decomposed_MSample))/2 - 1)/(2*np.pi) # Value in waves
    retardance = np.arccos(np.trace(r_decomposed_MSample)/2 - 1)/(2*np.pi) # Value in waves

    Retardance_Error = propagated_error(r_decomposed_MSample, RMS_Error)
    
    return MSample, retardance, MCal, RMS_Error, Retardance_Error 
