import numpy as np
import polutils as pol

def PSG(nsteps):

    theta = np.linspace(0,np.pi,nsteps)

    # Fixed Linear Polarizer
    polarizer = pol.hLinearPolarizer()
    psg = pol.JonesRotate(pol.QWavePlate(),theta)

    # Get the Analyzer Vector


def Polarimeter(jones):

    # Given a Jones Pupil, can we recover it?
    # Still not 100% certain as to what phi is

    pass
