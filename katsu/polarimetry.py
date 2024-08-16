from .mueller import linear_retarder, linear_polarizer, linear_diattenuator, wollaston
from .katsu_math import broadcast_kron, np
from scipy.optimize import curve_fit, minimize


def full_mueller_polarimetry(thetas, power, angular_increment,
                             return_condition_number = False,
                             Min = None,
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
    S0 = 2 * (a0 - a4)
    S1 = 4 * a4
    S2 = 4 * b4
    S3 = -2 * b2

    if return_coeffs:
        return np.array([S0, S1, S2, S3]), popt
    else:
        return np.array([S0, S1, S2, S3])

# TODO: Figure out a way to get the dual_channel_polarimetry_function to use
# this generalized function instead of separate sinusoidal fitting functions
def dual_channel_sinusoid(theta, I, Q, U, theta_2 = None, 
        method = "single_difference", normalized = False):
    """
    Calculate the sinusoidal response for a dual channel polarimetry setup.

    Parameters
    ----------
    theta : float or ndarray
        The angle(s) of the half-wave plate (HWP) in radians
    I : float
        Stokes I of the input Stokes vector
    Q : float
        Stokes Q of the input Stokes vector
    U : float
        Stokes U of the input Stokes vector
    theta_2 : float or ndarray, optional
        The angle(s) of the HWP for the previous measurement, required for 
        the double difference method
    method : str, optional
        The differencing method to use, either "single_difference" or 
        "double_difference". Default is "single_difference"
    normalized : bool, optional
        Whether the output should be normalized by the total intensity I. 
        Default is False

    Returns
    -------
    float or ndarray
        the sinusoidal power based on the provided Stokes parameters, 
        angles, and method

    Raises
    ------
    ValueError
        If `normalized=True` and `I` is not provided or if `theta_2` is 
        required and not provided.
    """

    """Sinusoidal response for dual channel polarimetry."""

    if method == "single_difference":
        if normalized:
            return (Q * np.cos(4 * theta) + U * np.sin(4 * theta)) / I
        else:
            return (Q * np.cos(4 * theta) + U * np.sin(4 * theta))
    elif method == "double_difference":
        if normalized:
            return (Q * (np.cos(4 * theta) - np.cos(4 * theta_2)) + \
                U * (np.sin(4 * theta) - np.sin(4 * theta_2)) / (2 * I))
        else:
            return (Q * (np.cos(4 * theta) - np.cos(4 * theta_2)) + \
                U * (np.sin(4 * theta) - np.sin(4 * theta_2)))
        
def unnormalized_single_diff_sinusoid(theta, Q, U):
    """
    Calculate the unnormalized sinusoidal response for a single difference in 
    dual channel polarimetry.

    Parameters
    ----------
    theta : float or ndarray
        The angle(s) of the half-wave plate (HWP) in radians
    Q : float
        Stokes Q of the input Stokes vector
    U : float
        Stokes U of the input Stokes vector

    Returns
    -------
    float or ndarray
        The output power response based on the provided Stokes parameters 
        and angles
    """
    output_power = (Q * np.cos(4 * theta) + U * np.sin(4 * theta))
    return output_power

def unnormalized_double_diff_sinusoid(theta_1, theta_2, Q, U):
    """
    Calculate the unnormalized sinusoidal response for a double difference 
    in dual channel polarimetry.

    Parameters
    ----------
    theta_1 : float or ndarray
        The angle(s) of the half-wave plate (HWP) in radians for the first 
        measurement
    theta_2 : float or ndarray
        The angle(s) of the HWP in radians for the second measurement
    Q : float
        Stokes Q of the input Stokes vector
    U : float
        Stokes U of the input Stokes vector

    Returns
    -------
    float or ndarray
        The output power response based on the provided Stokes parameters 
        and angles
    """

    output_power = (Q * (np.cos(4 * theta_2) - np.cos(4 * theta_1)) + \
        U * (np.sin(4 * theta_2) - np.sin(4 * theta_1)))
    return output_power

# TODO: Test the implentation of normalized single difference
def normalized_single_diff_sinusoid(theta, I, Q, U):
    """
    Calculate the normalized sinusoidal response for a single difference 
    in dual channel polarimetry

    Parameters
    ----------
    theta : float or ndarray
        The angle(s) of the half-wave plate (HWP) in radians
    Q : float
        Stokes Q of the input Stokes vector
    U : float
        Stokes U of the input Stokes vector

    Returns
    -------
    float or ndarray
        The normalized output power response based on the provided Stokes 
        parameters and angles.
    """
    
    output_power = (Q * np.cos(4 * theta) + U * np.sin(4 * theta) / I)
    return output_power

# TODO: Test the implentation of normalized double difference
def normalized_double_diff_sinusoid(theta_1, theta_2, I, Q, U):
    """
    Calculate the normalized sinusoidal response for a double difference 
    in dual channel polarimetry.

    Parameters
    ----------
    theta_1 : float or ndarray
        The angle(s) of the half-wave plate (HWP) in radians for the first 
        measurement
    theta_2 : float or ndarray
        The angle(s) of the HWP in radians for the second measurement.
    I : float
        Stokes I of the input Stokes vector
    Q : float
        Stokes Q of the input Stokes vector
    U : float
        Stokes U of the input Stokes vector

    Returns
    -------
    float or ndarray
        The normalized output power response based on the provided Stokes 
        parameters and angles
    """

    output_power = (Q * (np.cos(4 * theta_1) - np.cos(4 * theta_2)) + \
        U * (np.sin(4 * theta_1) - np.sin(4 * theta_2)) / (2 * I))
    return output_power

def dual_channel_polarimeter(thetas, S_in = None, power_o = None, 
        power_e = None, normalized = False, sub_method = "single_difference"):
    """
    Simulate or analyze a dual channel polarimetry experiment using single 
    or double differencing.

    Parameters
    ----------
    thetas : ndarray
        Array of angles (in radians) at which measurements were taken
    S_in : ndarray, optional
        The input Stokes vector [I, Q, U, V] to be measured. Default is None.
    power_o : ndarray, optional
        Measured power in the ordinary beam. Default is None.
    power_e : ndarray, optional
        Measured power in the extraordinary beam. Default is None.
    normalized : bool, optional
        Whether to normalize the response by the total intensity. 
        Default is False
    sub_method : str, optional
        The differencing method to use, either "single_difference" or 
        "double_difference". Default is "single_difference"

    Returns
    -------
    ndarray
        The fitted or propagated Stokes components [Q, U] based on the 
        differencing and normalization method specified.

    Raises
    ------
    ValueError
        If required input data is missing or if invalid parameters are 
        provided.
    """

    # Empty arrays for HWP angles and measured power
    nmeas = len(thetas)
    Pmat = np.zeros(nmeas)

    # Empty arrays for Mueller matrices of the ordinary and extraordinary beams
    M_o_beams = np.zeros((nmeas, 4, 4))
    M_e_beams = np.zeros((nmeas, 4, 4))

    for i in range(nmeas):
        # Use the provided wollaston and linear_retarder functions
        M_wollaston_o = wollaston(beam = 0)
        M_wollaston_e = wollaston(beam = 1)

        # Combined Mueller matrix of HWP and Wollaston prism
        M_o_beams[i] = M_wollaston_o @ linear_retarder(thetas[i], np.pi)
        M_e_beams[i] = M_wollaston_e @ linear_retarder(thetas[i], np.pi)

        # Extracting power measurements to compute single and double differences
        # or propagating input Stokes vector through the system
        if power_o is not None and power_e is not None:
            Pmat_o_current = power_o[i]
            Pmat_e_current = power_e[i]
            if i > 0:
                Pmat_o_previous = power_o[i - 1]
                Pmat_e_previous = power_e[i - 1]
        elif S_in is not None:
            Pmat_o_current = np.dot(M_o_beams[i][0, :], S_in)
            Pmat_e_current = np.dot(M_e_beams[i][0, :], S_in)
            if i > 0:
                Pmat_o_previous = np.dot(M_o_beams[i - 1][0, :], S_in)
                Pmat_e_previous = np.dot(M_e_beams[i - 1][0, :], S_in)

        # Computing single and double differences
        if sub_method == "single_difference":
            if normalized:
                Pmat[i] = (Pmat_o_current - Pmat_e_current) / \
                    (Pmat_o_current + Pmat_e_current)
            else:
                Pmat[i] = Pmat_o_current - Pmat_e_current
        elif sub_method == "double_difference":
            if i == 0:
                # Keep first measurement as is
                if normalized:
                    Pmat[i] = (Pmat_o_current - Pmat_e_current) / \
                        (Pmat_o_current + Pmat_e_current)
                else:
                    Pmat[i] = Pmat_o_current - Pmat_e_current  
            else:
                if normalized:
                    Pmat[i] = ((Pmat_o_current - Pmat_e_current) - \
                        (Pmat_o_previous - Pmat_e_previous)) / \
                        ((Pmat_o_current + Pmat_e_current) + \
                        (Pmat_o_previous + Pmat_e_previous))
                else:
                    Pmat[i] = (Pmat_o_current - Pmat_e_current) - \
                        (Pmat_o_previous - Pmat_e_previous)

    # Fitting for Stokes Q and U for unnormalized single difference
    if sub_method == "single_difference" and not normalized:
        objective = lambda params: \
            np.sum((unnormalized_single_diff_sinusoid(thetas, *params) - Pmat) ** 2)
        
        initial_guess = [0, 0]

        result = minimize(objective, initial_guess)

        Q_fit, U_fit = result.x

        return np.array([1, Q_fit, U_fit, 0])
    # Fitting for Stokes Q and U for normalized single difference
    elif sub_method == "double_difference" and not normalized:
        objective = lambda params: \
            np.sum((unnormalized_double_diff_sinusoid(thetas[ : -1], 
                thetas[1 : ], *params) - Pmat[1 : ]) ** 2)
        
        initial_guess = [0, 0]

        result = minimize(objective, initial_guess)

        Q_fit, U_fit = result.x

        return np.array([1, Q_fit, U_fit, 0])

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
        return M, condition_number(Wmat)

    else:
        return M