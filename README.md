# Katsu: Integrated polarimetry and polarization simulation
[![codecov](https://codecov.io/gh/Jashcraf/katsu/graph/badge.svg?token=NXLEQE61YX)](https://codecov.io/gh/Jashcraf/katsu)

[![Documentation Status](https://readthedocs.org/projects/katsu/badge/?version=latest)](https://katsu.readthedocs.io/en/latest/?badge=latest)

Katsu is a Python 3.8+ library that contains an integrated library for modeling simple polarization effects (represented with Mueller calculus), simulating full stokes and mueller polarimetry, and integrating both of these as data reduction tools for conducting polarimetry in the laboratory. We also feature motion control routines for commercially available rotation stages for a more Pythonic interface to devices that would otherwise require serial communication.

## Installation
Currently Katsu is installable from source, just run the following in your terminal
```
git clone https://github.com/Jashcraf/katsu/
cd katsu
pip install .
```

## Acknowledgements
Big thanks to Quinn Jarecki of UA's Polarization Lab for the starting theory on dual rotating retarder mueller polarimetry, and for overall helpful discussions. Thanks also to William Melby, Manxuan Zhang, and Max Millar-Blanchaer for being the first to test out the Mueller data reduction code.
