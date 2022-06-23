import numpy as np
import polutils as pol
import mueller as mul
import matplotlib.pyplot as plt
from numba import njit

def GenerateStokesArray(svector,npix):

    polarray = np.ones([npix,npix,4])
    polarray[:,:,0] = svector[0]
    polarray[:,:,1] = svector[1]
    polarray[:,:,2] = svector[2]
    polarray[:,:,3] = svector[3]

    return polarray

def GenerateMuellerMatrixArray(mueller,npix):
    """
    npix is raveled npix, for a square grid of side dimension npix. the npix here = npix**2
    """

    marray = np.ones([4,4,npix])
    marray *= mueller

def PropagateThroughMuellerSystem(sarray,Msys):
    sout = Msys @ sarray
    return sout

def StokesAnalyzer(analyzervector,stokes):
    return np.dot(analyzervector,stokes)

def NoisyStokesAnalyzer(analyzervector,stokes):

    # Poisson Distribution noise?
    return np.dot(analyzervector,stokes) + np.random.poisson(lam=1)*1e-2

def DualTetrahedronPolarizations():

    v1 = np.array([1,1,0,-1/np.sqrt(2)])
    v2 = np.array([1,-1,0,-1/np.sqrt(2)])
    v3 = np.array([1,0,1,1/np.sqrt(2)])
    v4 = np.array([1,0,-1,1/np.sqrt(2)])

    # v1 = np.array([1,1,0,0])
    # v2 = np.array([1,-1/3,2*np.sqrt(3)/3,0])
    # v3 = np.array([1,-1/3,-np.sqrt(2)/3,np.sqrt(2/3)])
    # v4 = np.array([1,-1/3,-np.sqrt(2)/3,-np.sqrt(2/3)])

    # Check uniqueness

    return v1,v2,v3,v4

def StokesSinusoid(nmeas,a0,b2,a4,b4):
    return a0 + b2*np.sin(2*nmeas) + a4*np.cos(4*nmeas) + b4*np.sin(4*nmeas)

def MuellerSinusoid(nmeas,
                    a0,a2,a3,a4,a6,a7,a8,a9,a10,a11,a12,
                    b1,b2,b3,b5,b7,b8,b9,b10,b11,b12):

    signal = a0
    signal += a2*np.cos(2*nmeas)
    signal += a3*np.cos(3*nmeas)
    signal += a4*np.cos(4*nmeas)
    signal += a6*np.cos(6*nmeas)
    signal += a7*np.cos(7*nmeas)
    signal += a8*np.cos(8*nmeas)
    signal += a9*np.cos(9*nmeas)
    signal += a10*np.cos(10*nmeas)
    signal += a11*np.cos(11*nmeas)
    signal += a12*np.cos(12*nmeas)

    signal += b1*np.sin(1*nmeas)
    signal += b2*np.sin(2*nmeas)
    signal += b3*np.sin(3*nmeas)
    signal += b5*np.sin(5*nmeas)
    signal += b7*np.sin(7*nmeas)
    signal += b8*np.sin(8*nmeas)
    signal += b9*np.sin(9*nmeas)
    signal += b10*np.sin(10*nmeas)
    signal += b11*np.sin(11*nmeas)
    signal += b12*np.sin(12*nmeas)

    return signal

def FullMuellerPolarimeterMeasurement(Min,nmeas):

    from scipy.optimize import curve_fit
    Wmat = np.zeros([16,nmeas])
    Pmat = np.zeros([nmeas])
    wcount = 0
    th = np.linspace(0,np.pi,nmeas)

    # Quinn's Offsets
    thg = 0# -17.02*np.pi/180
    tha = 0 #13.41*np.pi/180
    thlp = 0# -0.53*np.pi/180

    for i in range(nmeas):

        # Mueller Matrix of Generator
        Mg = mul.LinearRetarder(th[i] + thg,np.pi/2) @ mul.LinearPolarizer(0)

        # Mueller Matrix of Analyzer
        Ma = mul.LinearPolarizer(thlp) @ mul.LinearRetarder(th[i]*5 + tha,np.pi/2)

        # Mueller Matrix of System and Generator
        # A detector measures the first row of the analyzer matrix
        Wmat[:,i] = np.kron(Ma[0,:],Mg[:,0])
        # print('Analyzer = ',Ma[0,:])
        # print('Generator = ',Mg[0,:])
        Pmat[i] = Ma[0,:] @ Min @ Mg[:,0]

    popt,pcov = curve_fit(MuellerSinusoid,
                          th,
                          Pmat)

    Malt = Pmat @ np.linalg.pinv(Wmat)

    # The even coefficients
    a0 = popt[0]
    a2 = popt[1]
    a3 = popt[2]
    a4 = popt[3]
    a6 = popt[4]
    a7 = popt[5]
    a8 = popt[6]
    a9 = popt[7]
    a10 = popt[8]
    a11 = popt[9]
    a12 = popt[10]

    # The odd coefficients
    b1 = popt[11]
    b2 = popt[12]
    b3 = popt[13]
    b5 = popt[14]
    b7 = popt[15]
    b8 = popt[16]
    b9 = popt[17]
    b10 = popt[18]
    b11 = popt[19]
    b12 = popt[20]

    sineval = MuellerSinusoid(th,a0,a2,a3,a4,a6,a7,a8,a9,a10,a11,a12,b1,b2,b3,b5,b7,b8,b9,b10,b11,b12)
    print('Sinusoid Shape')
    print(sineval.shape)

    plt.figure()
    plt.title('Irradiance Measurements')
    plt.plot(th,sineval,label='sinusoid fit',linestyle='dashdot')
    plt.plot(th,Pmat,label='Power measurement')
    plt.legend()
    plt.xlabel('Measurement')
    plt.show()

    # The Mueller Matrix Elements
    m44 = a6-a4
    m43 = 2*(a7-a3)
    m42 = -2*(b3+b7)
    m41 = -(b5 + m42/2)
    
    m34 = 2*(a9-a11)
    m33 = 4*(a8-a12)
    m32 = 4*(b8+b12)
    m31 = 2*b10 - m32/2

    m24 = 2*(b11-b9)
    m23 = 4*(-b8 + b12)
    m22 = 4*(a8+a12)
    m21 = 2*a10 - m22/2

    m14 = b1 - m24/2
    m13 = 2*b2 - m23/2
    m12 = 2*a2 - m22/2
    m11 = a0 - m12/2 - m21/2 - m22/4

    M = np.array([[m11,m12,m13,m14],
                  [m21,m22,m23,m24],
                  [m31,m32,m33,m34],
                  [m41,m42,m43,m44]])

    return M,Malt

def FullStokesPolarimeterMeasurement(Sin,nmeas):

    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    
    # This is a rotating retarder mueller matrix
    # Wmat = np.zeros([4,nmeas])
    Pmat = np.zeros([nmeas])
    # wcount = 0

    # Retarder needs to rotate 2pi, break up by nmeas
    th = np.linspace(0,2*np.pi,nmeas)
    thp = np.linspace(0,2*np.pi,101)

    for i in range(nmeas):

        # Mueller Matrix of analyzer
        M = mul.LinearPolarizer(0) @ mul.LinearRetarder(th[i],np.pi/2)

        # The top row is the analyzer vector
        analyzer = M[0,0:4]

        # Record the power
        Pmat[i] = NoisyStokesAnalyzer(analyzer,Sin)

        # Optionally, add some photon noise


    popt,pcov = curve_fit(StokesSinusoid,
                          th,
                          Pmat,
                          p0 = (1,1,1,1))

    a0 = popt[0]
    b2 = popt[1]
    a4 = popt[2]
    b4 = popt[3]

    sineval = StokesSinusoid(thp,a0,b2,a4,b4)

    # plt.figure()
    # plt.title('Flux Measurements')
    # plt.scatter(th,Pmat,label='Irradiance Measurements',marker='*')
    # plt.plot(thp,sineval,linestyle='dashdot',label='Curve-fit Sinusoid')
    # plt.legend()
    # plt.show()

    # Compute the Stokes Vector
    S0 = 2*(a0 - a4)
    S1 = 4*a4
    S2 = 4*b4
    S3 = -2*b2

    return np.array([S0,S1,S2,S3])

def FullStokesPolarimeter(Sarray,nmeas):

    dim = Sarray.shape[0]
    Sout = np.zeros(Sarray.shape)

    for ijk in range(dim):
        for lmn in range(dim):

            Sin = Sarray[ijk,lmn,:]

            Sout[ijk,lmn,:] = FullStokesPolarimeterMeasurement(Sin,nmeas)

    return Sout

def PlotStokesArray(Sarray,Sin=None):

    dim = Sarray.shape[0]

    x = np.linspace(-1,1,dim)
    x,y = np.meshgrid(x,x)

    vmin = -1
    vmax = 1

    plt.figure(figsize=[20,5])
    plt.suptitle('Recovered Stokes Image')
    plt.subplot(141)
    plt.title('S0')
    plt.imshow(Sarray[:,:,0],vmin=vmin,vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(142)
    plt.title('S1')
    plt.imshow(Sarray[:,:,1],vmin=vmin,vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(143)
    plt.title('S2')
    plt.imshow(Sarray[:,:,2],vmin=vmin,vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(144)
    plt.title('S3')
    plt.imshow(Sarray[:,:,3],vmin=vmin,vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    if type(Sin) == np.ndarray:

        vmin = -1e-1
        vmax = 1e-1

        plt.figure(figsize=[20,5])
        plt.suptitle('Difference from Nominal State')
        plt.subplot(141)
        plt.title('S0')
        plt.imshow(Sarray[:,:,0]-Sin[:,:,0],vmin=vmin,vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(142)
        plt.title('S1')
        plt.imshow(Sarray[:,:,1]-Sin[:,:,1],vmin=vmin,vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(143)
        plt.title('S2')
        plt.imshow(Sarray[:,:,2]-Sin[:,:,2],vmin=vmin,vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(144)
        plt.title('S3')
        plt.imshow(Sarray[:,:,3]-Sin[:,:,3],vmin=vmin,vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

def AddNoise(fluxmap):
    return




def DualTetrahedronPolarimeter(mueller):

    print('Mueller Matrix to Reconstruct')
    print(mueller)

    v1,v2,v3,v4 = DualTetrahedronPolarizations()

    polstates = [v1,v2,v3,v4]

    # Test np.ravel for mueller matrices
    M00 = mueller[0,0]
    M01 = mueller[0,1]
    M02 = mueller[0,2]
    M03 = mueller[0,3]

    M10 = mueller[1,0]
    M11 = mueller[1,1]
    M12 = mueller[1,2]
    M13 = mueller[1,3]

    M20 = mueller[2,0]
    M21 = mueller[2,1]
    M22 = mueller[2,2]
    M23 = mueller[2,3]

    M30 = mueller[3,0]
    M31 = mueller[3,1]
    M32 = mueller[3,2]
    M33 = mueller[3,3]

    Wmat = np.zeros([16,16])
    Pmat = np.zeros([16])
    wcount = 0

    for ijk in range(4): # the analyzer
        for lmn in range(4): # the source

            # Separate 16 Mueller Elements
            input = polstates[lmn]
            analyzer = polstates[ijk]

            # Generate Final Power Measurement
            P = analyzer @ mueller @ np.transpose(input)

            # Polarimetric Data Reduction Matrix Element
            # Wmat[wcount,0] = M00*input[0]*analyzer[0]
            # Wmat[wcount,1] = M01*input[1]*analyzer[1]
            # Wmat[wcount,2] = M02*input[2]*analyzer[2]
            # Wmat[wcount,3] = M03*input[3]*analyzer[3]

            # Wmat[wcount,4] = M10*input[0]*analyzer[0]
            # Wmat[wcount,5] = M11*input[1]*analyzer[1]
            # Wmat[wcount,6] = M12*input[2]*analyzer[2]
            # Wmat[wcount,7] = M13*input[3]*analyzer[3]

            # Wmat[wcount,8] = M20*input[0]*analyzer[0]
            # Wmat[wcount,9] = M21*input[1]*analyzer[1]
            # Wmat[wcount,10] = M22*input[2]*analyzer[2]
            # Wmat[wcount,11] = M23*input[3]*analyzer[3]

            # Wmat[wcount,12] = M30*input[0]*analyzer[0]
            # Wmat[wcount,13] = M31*input[1]*analyzer[1]
            # Wmat[wcount,14] = M32*input[2]*analyzer[2]
            # Wmat[wcount,15] = M33*input[3]*analyzer[3]

            # Try the Kronecker Product
            Wmat[wcount,:] = np.kron(analyzer,input)

            Pmat[wcount] = P
            wcount += 1

    print('rank=',np.linalg.matrix_rank((Wmat)))
    print(Wmat)
    Mout = np.linalg.pinv(Wmat) @ np.transpose(Pmat)

    return Mout


    

    



    # v1a = v1
    # v2a = v2
    # v3a = v3
    # v4a = v4 

    # v1in = np.transpose(v1)
    # v2in = np.transpose(v2)
    # v3in = np.transpose(v3)
    # v4in = np.transpose(v4)

    # v1out = mueller @ v1in
    # v2out = mueller @ v2in
    # v3out = mueller @ v3in
    # v4out = mueller @ v4in
    
    # P1 = np.dot(v1,v1out)
    # P2 = np.dot(v1,v2out)
    # P3 = np.dot(v1,v3out)
    # P4 = np.dot(v1,v4out)

    # P5 = np.dot(v2,v1out)

def ConstructPolReductionMatrix(measurements):



    return




    