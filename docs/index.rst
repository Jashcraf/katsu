.. Katsu documentation master file, created by
   sphinx-quickstart on Fri Mar  8 10:46:54 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Katsu's documentation!
=================================
Katsu is a package for enabling polarimetry in the laboratory, complete with capabilities for polarimeter simulation, polarimetric data reduction routines, and some simple classes for serial communication with rotation stages that enable polarimetry. Katsu was developed as a part of a NASA Space Technology Graduate Research Opportunity Visiting Technologist Experience at Subaru Telescope, with the aim of enhancing our understanding of instrumental polarization in astronomical telescopes.

Summary
-------

**What this software does:**

* Spatially-broadcasted Mueller calculus: Katsu is structured to enable 
* Routines for full Stokes and Mueller data reduction: Katsu contains routines for polarimetric data reduction for full Stokes and Mueller polarimeters
* Options for controlling motion stages that enable polarimetry: Katsu contains a full Python API for Newport's Agilis Rotation mounts, enabling the operation of rotating-retarder polarimeters in the laboratory.

.. contents::
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   modules

.. toctree::
   :maxdepth: 2
   :caption: Community Guidelines

   notebooks/CommunityGuidelines.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Installation

   notebooks/Installation.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/tutorials/MuellerCalculus.ipynb
   notebooks/tutorials/PolarDecomposition.ipynb

.. toctree::
   :maxdepth: 2
   :caption: How-Tos

   notebooks/howtos/FullStokesExample.ipynb
   notebooks/howtos/FullMuellerExample.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Explanations

   notebooks/explanation/BroadcastedPolarimetry.ipynb
   notebooks/explanation/DemoAgilisMotion.ipynb
   notebooks/explanation/DualChannelPolarimetryIntro.ipynb
   notebooks/explanation/Q_DataReduction.ipynb



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
