---
title: 'Katsu: A Python package for Mueller and Stokes simulation and polarimetry'
tags:
  - Python
  - astronomy
  - polarimetry
  - laboratory
  - mueller
  - stokes
authors:
  - name: Jaren N. Ashcraft
    orcid: 0000-0001-5082-7442
    corresponding: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Ewan S. Douglas
    orcid: 0000-0002-0813-4308
    affiliation: 2
  - name: William Melby
    orcid: 0000-0001-5178-689X
    affiliation: 3
  - name: Manxuan Zhang
    orcid: 0000-0003-3567-6839
    affiliation: 3
  - name: Kenji Mulhall
    affiliation: 4
  - name: Ramya M. Anche
    orcid: 0000-0002-4989-6253
    affiliation: 2
  - name: Emory Jenkins
    affiliation: "1, 2"
    orcid: 0000-0001-6829-7910
  - name: Maxwell A. Millar-Blanchaer
    orcid: 0000-0001-6205-9233
    affiliation: 3
  
affiliations:
 - name: Wyant College of Optical Sciences, University of Arizona, USA
   index: 1
 - name: Steward Observatory, University of Arizona, USA
   index: 2
 - name: Department of Physics, University of California, Santa Barbara, USA
   index: 3
 - name: Independent contributor
   index: 4
date: 16 August 2024
bibliography: paper.bib
---

# Summary
High-performance simulation of physical optics phenomena is instrumental to accurately design and understand optical systems. The propagation of light can be described in multiple regimes. The geometrical regime treats light as a ray, enabling the expeditious optimization of optical surfaces. Pythonic examples in the open source include Hayford et al.'s `ray-optics` [@ray-optics] and Myers et al.'s `batoid` [@batoid]. The physical regime treats light as a wave, enabling a precise understanding of the field distribution as light propagates through a given system. Many Python-based packages exist for wave optics propagation, including `POPPY` [@poppy], `HCIPy` [@hcipy], `prysm` [@Dube2019;@Dube:22], and `dLux` [@dLux]. However, all of the aforementioned packages treat light as a scalar field, and are unable to simulate the propagation of the vector nature of light. This property, called _polarization_, is critical for various instruments that interact with the vector nature of light.

To describe the propagation of light's polarization state, we can use Mueller calculus. This approach represents optical systems as a 4 $\times$ 4 _Mueller Matrix_, ($\mathbf{M}$) that operate on a _Stokes Vector_  ($\mathbf{s}$) which represents the polarization of light, as shown in the Equation below,

$$\mathbf{s}' = \mathbf{M} \mathbf{s}.$$

The Stokes vector contains the parameters used to describe the polarization of light,

$$
\mathbf{s} = [s_{0}, s_{1}, s_{2}, s_{3}],
$$

where $s_{0}$ represents the unpolarized intensity, $s_{1}$ describes the degree of polarization oriented along $0^{\circ} / 90^{\circ}$, $s_{2}$ describes the degree of polarization oriented along $\pm 45^{\circ}$, and $s_{3}$ describes the degree of circular polarization. The Stokes parameters are equivalently represented with the letter notation,

$$
\mathbf{s} = [I, Q, U, V].
$$

Mueller calculus is particularly powerful because it is capable of describing light that is partially polarized, and the Stokes parameters are quantities that can be easily measured in the laboratory. 

`Katsu` is an open-source Python package to address the need for polarimetric characterization of astronomical systems for the next generation of astronomical telescopes. It contains simple routines for the simulation of Mueller matrices and Stokes vectors to model how polarization is transformed by optical systems. This ability is not unique to `Katsu`; another package capable of simple Mueller calculus is the Pypolar [@pypol] package by Prahl, which also contains excellent visualization tools as well as support for ellipsometric data reduction. However, one area where `Katsu` is distinct from other packages capable of Mueller calculus simulation is its emphasis on broadcasted matrix calculations. All Mueller matrices available in `katsu.mueller` take a `shape` keword that append an arbitrary number of axes to the front of the initialized array, with the final two axes containing the Mueller matrix. This functionality is critical for accelerated computating on spatial data, which enables the direct measurement of polarization aberrations in the lab [@ashcraft2024]. `Katsu` also features a `polarimetry` module containing the data reduction routines for single-rotating retarder (SRR) Stokes polarimeters and dual-rotating retarder (DRR) Mueller polarimeters. In the pursuit of open-source instrumentation in the laboratory, `Katsu` supports an interface to the Newport Agilis series rotation mounts in the `motion` module to assist with performing polarimetry with rotating retarders. 

To interpret the measurements made in the lab, `Katsu` features the polar decomposition of Mueller matrices published by Lu and Chipman [@Lu:96]. This decomposes a Mueller matrix $\mathbf{M}$ into its constituent depolarizer $\mathbf{M}_{\Delta}$, diattenuator $\mathbf{M_{D}}$, and retarder $\mathbf{M_{R}}$, as shown in the following Equation,

$$\mathbf{M} = \mathbf{M}_{\Delta}\mathbf{M_{R}}\mathbf{M_{D}}. $$

This function is critical for separating depolarization from the Mueller matrix, which enables the integration of polarization aberration in the laboratory into simulated data (generated by a polarization ray tracer, e.g. `Poke` [@Ashcraft_2023], `Pyastropol` [@Pruthvi.2020]). `Katsu` also adopts the flexible interchangeable backend system of `prysm` [@Dube2019;@Dube:22] for hot-swappable `numpy`-like backends [@harris2020array] including `cupy` for GPU-accelerated computing [@cupy_learningsys2017] and `jax` for automatic differentiation [@jax2018github].

# Statement of need
Polarimetric characterization in the laboratory is critical to the performance of astronomical instrumentation. The next generation of astronomical observatories has identified that wavefront errors induced by polarization, called _polarization aberrations_, could be a limiting factor in direct exoplanet imaging. Ample modeling has been done at the observatory level to understand the nominal polarization aberrations that arise in the telescope [@anche2023polarization;@gaudi2020habitable;@Will_polarization_luvoir] but less work has been done to characterize instrumentation in the laboratory.

`Katsu` has recently been used as the primary backend of the Gromit polarimeter at the UA Space Astrophysics Laboratory [@gromit], and used to measure the spatially-varying polarization aberrations of the Space Coronagraph Optical Bench (SCoOB) [@ashcraft2024]. The measurement and polarimetric data reduction capabilities available in `Katsu` enabled expeditious full Mueller polarimetry of an astronomical coronagraph testbed. In the future, we aim to add more polarimetric data reduction capabilities to `Katsu`, like a recently-published modification of DRRP data reduction to leverage insights from dual-channel polarimetry [@melby2024] for passive insensitivity to variations in total intensity. 

# Acknowledgements
This work was sponsored by a NASA Space Technology Graduate Research Opportunity. J.N.A. acknowledges support by NASA through the NASA Hubble Fellowship grant #HST-HF2-51547.001-A awarded by the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy.

# References