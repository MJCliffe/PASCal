---
  title: 'PASCal Python: A Principal Axis Strain Calculator'
  tags:
  - Python
  - crystallography
  - non-ambient
  - thermal expansion
  - compressibility
  - electrochemical strain

  authors:
    - name: Monthakan Lertkiattrakul
      affiliation: 1
    - name: Matthew L. Evans
      orcid:  0000-0002-1182-9098
      affiliation: 2
    - name: Matthew J. Cliffe
      orcid: 0000-0002-0408-7647
      affiliation: 1
  affiliations:
  - name: School of Chemistry, University Park, Nottingham, NG7 2RD, United Kingdom
    index: 1
  - name: Institut de la Matière Condensée et des Nanosciences, Université catholique de Louvain, Chemin des Étoiles 8, Louvain-la-Neuve 1348, Belgium
    index: 2
  date: 28 May 2023

  bibliography: paper.bib
...

# Summary

The response of crystalline materials to external stimuli: whether temperature, pressure or electrochemical potential, is critical for both our understanding of materials and their use. This information can be readily obtained through in-situ diffraction experiments, however if the intrinsic anisotropy of crystals is not taken into account, the true behaviour of crystals can be overlooked. This is particularly true for anomalous mechanical properties of great topical interest, such as negative linear or area compressibility [@Cairns2015; @Hodgson2014], negative thermal expansion [@chenNegativeThermalExpansion2015] or strongly anisotropic electrochemical strain [@kondrakovAnisotropicLatticeStrain2017].

We have developed PASCal, Principal Axis Strain Calculator, a widely used web tool (https://github.com/MJCliffe/PASCal) that implements the rapid calculation of principal strains and fitting to many common models for equations of state. It provides a simple web form user interface designed to be able to be used by all levels of experience. This new version of PASCal is written in Python using the standard scientific Python stack [@harrisArrayProgrammingNumPy2020; @virtanenSciPyFundamentalAlgorithms2020], is released open source under the MIT license, and significantly extends the feature set of the original closed-source Fortran, Perl and Gnuplot webtool [@Cliffe2012b]. Significant additional attention has been paid to testing, documentation, modularisation and reproducibility, enabling the main app functionality to now also be accessed directly through a Python API. The web app is deployed online at [https://www.pascalapp.co.uk](https://www.pascalapp.co.uk) with the associated source code and documentation available on GitHub at [MJCliffe/PASCal](https://github.com/MJCliffe/PASCal).


# Statement of Need

Characterising the anisotropic strain response of a crystalline material requires calculating not just the deformation along unit cell axes directions, which for low symmetry crystals have an arbitrary orientation relative to the mechanical properties, but also how changes in unit cell angles affect the principal (eigen-)strains and their orientations (eigenvectors). A great deal of additional information can be extracted from the stimulus-dependence of these principal strains, and fitting this dependence to equations-of-state can produce parameters of fundamental importance (e.g., compressibilities). Software capable of transforming experimental lattice parameters into the principal strains and the derived underlying thermodynamic properties (e.g. expansivities or compressibilities) is therefore of great utility to experimental scientists studying the response of crystals.

PASCal is designed to be user-friendly and so accessible for the broader materials chemistry community: with a simple plain-text input, a limited number of options for each data-type, and easy to read and export output. It can calculate strains for any dataset containing crystallographic unit cell data as a function of an external variable, but has three standard options for variable temperature, pressure and electrochemical data, which selects the appropriate units and equations of state. This new version of PASCal (v2.0) introduces a number of new features including more accurate eigenvector matching, capabilities for electrochemical data, and more robust fitting to equation-of-state.

# Description of the software

PASCal takes unit cell parameters (in standard $a$, $b$, $c$, $\alpha$, $\beta$, $\gamma$ format), as a function of an external stimulus (temperature, pressure or electrochemical charge) as input.

The strain is calculated from the unit cell parameters in the conventional fashion [@Giacovazzo2011]. First, the unit cell parameters for each data point are orthogonalised using the Institute of Radio Engineers convention, where $\mathbf{z}$ is parallel to the $\mathbf{c}$ crystallographic axis, $\mathbf{x}$ is parallel to $\mathbf{a}^\ast$, and $\mathbf{y} = \mathbf{z} \times \mathbf{x}$. The strain, $\epsilon$, is calculated from the symmetrised change between the initial data point and the each data point (Lagrangian) or between each data point and the initial data point (Eulerian) with options for either finite or infinitesimal strain.

The strain matrix is then diagonalised, to yield the principal strains (eigenvalues), and their directions (eigenvectors). In general, the ordering of the principal axes from any diagonalisation algorithm will not remain constant over the data series, particularly where the underlying data are noisy. The axes are therefore matched such that the directions remain consistent with those of the first data points.

These matched and sorted strain eigenvalues are then fitted to equations of state appropriate to the data type. All data types are first fitted linearly. Pressure datasets, which are highly non-linear, are additionally fitted to the widely used empirical equation-of-state: $\epsilon(p) = \epsilon_0 + \lambda (p - p_c)^\nu$ [@Goodwin2008a]. For electrochemical datasets, where datasets are non-linear but analytic equations-of-state are not widely used, the data are fitted by Chebyshev polynomials with an appropriate regularisation procedure.

Linear fits are carried out on volumetric data, with additional fitting carried out for pressure datasets to the Birch-Murnaghan equations of state (second order, third order, and third order with critical pressure correction) [@Birch1947; @Sata2002] and for electrochemical datasets using Chebyshev polynomials. The errors in calculated parameters and their derivatives with respect to the stimuli (i.e., compressibilities) are calculated using the residuals of the fits, as propagation of crystallographic errors through to the strain is not in general well-defined. Errors in the stimulus are used solely to weight the relative importance of data points.

The output page consists of a number of tables of physically meaningful coefficients and interactive graphics [@plotlytechnologiesincCollaborativeDataScience2015] showing the stimuli dependence of the principal strains, their derivatives (for pressure and electrochemical data), and an indicatrix represention of the expansivity/compressibility.

# Comparison with other software

There are a number of other software packages that can calculate principal strains and fit equations of state. [EOSfit](http://www.rossangel.com/) is a suite of programs designed to calculate thermal expansion and equations of state as a function of pressure and recent versions include principal strain calculation also. Standalone calculation of strain is available in the no-longer maintained WinStrain. EOSfit is an offline, closed source package that has a GUI and command line interface. [STRAIN](https://www.cryst.ehu.es/cryst/strain.html) is a closed-source online tool included on the Bilbao Crystallographic Server, and can calculate strain calculations for a single pair of lattice parameters. [ELATE](https://progs.coudert.name/elate) is not designed for calculation of strains, but is a useful open-source tool for analysing full elastic constant tensors.[@ELATE2016]

# Acknowledgements

M.L. and M.J.C. acknowledge support from the Hobday bequest to the School of Chemistry, University of Nottingham and the University of Nottingham Propulsion Futures Beacon. M.L.E. thanks the BEWARE scheme of the Wallonia-Brussels Federation for funding under the European Commission's Marie Curie-Skłodowska Action (COFUND 847587). All authors also thank Ross Shonfield for help deploying the app and Madeleine Geers, Iain Oswald and Emily Meekel for their help testing the algorithms.

# References
