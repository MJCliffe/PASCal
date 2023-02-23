# PASCal

Principal Axis Strain Calculator (PASCal) is a web tool designed to help scientists analyse non-ambient lattice parameter data. It is written entirely in Python, using plotly to visualise the data, and as a web tool the code is designed to be used online, though it can be used offline with a local flask instance or by adapting the code for your own applications. The web tool is available at https://pascal-notts.azurewebsites.net, the code at https://github.com/MJCliffe/PASCal.

A paper summarising the original motivation and theory behind PASCal is published at [J. Appl. Cryst. (2012). 45, 1321-1329](https://doi.org/10.1107/S0021889812043026) ([arXiv](http://arxiv.org/pdf/1204.3007.pdf)). Please cite this if you use PASCal in a publication. This publication was produced for a previous version of PASCal and so description of the software itself is out of date. PASCal is designed might not be the most appropriate method for full analysis of your data, so please read the sections below on [errors](#errors), [fitting](#fitting) and [strains](#strain-calculation) if you have particular needs.

# Quick start

PASCal takes experimentally determined lattice parameters measured as a function of a parameter (temperature, pressure or charge), calculates the strain matrix, diagonalises this matrix to find the principal strains, and then carries out fits to simple functions to get summary metrics. Paste your values in, click calculate and see the results! There are helpful tips on hover, but more detailed information is below.

# Input 
Data should be input as plain text, with eight values per row, the control parameter $X$, its error $\sigma(X)$ and the lattice parameters (in Å and $^\circ$): X $\sigma (X)$ $a$ $b$ $c$ $\alpha$ $\beta$ $\gamma$. 

Lines beginning with # are comments and ignored: 
`#This line is metadata`.

There are a number of other options: 
- Data type: whether the data vary with temperature, pressure or electrochemical state-of-charge. This does not alter how the principal strains are calculated, but rather the fits (and units) that are carried out.
- For pressure data, whether this is a high pressure phase, and if so at what critical pressure it undergoes the transition.
- For electrochemical data, the maximum degree of Chebyshev polynomial to use to fit the lattice parameter and volume data.
- Advanced strain options. This determines which strain formalism is used to calculate the strain. 

# Output
## Summary Table
The summary table shows the three principal axes (in crystallographic coordinates [UVW]) and the coefficient of thermal expansion/compressibility/electrochemical strain derivative along each of these directions with the errors (see [below](#errors) for details). The coefficients are determined through fitting: using the linear fit for temperature, empirical fit for pressure and the Chebyshev fit for electrochemical strain. The principal axis shown is for the median data point, with the full list below.

## Plots
Below the summary tables are the key plots. These interactive plots are generated using plotly. A high resolution png can be saved using the camera button. Any given data series can be hidden by clicking on its symbol in the key and the plot. The raw data for each plot is shown in tables below (including the fitted values) if you wish to replot using different software. The indicatrix is a plot of the expansivity/compressibility tensor, showing its value along each direction. Positive values are blue, negative values are red and exact values can be measured using mouseover.

Plots are (data type):
1. Principal strains as a function of parameter with fit (TPX)
2. Derivative of fitted strain as a function of parameter (PX)
3. Volume (TPX)
4. Fit as a function of degree of polynomial (X)
5. Indicatrix (TPX)

## Tables
The raw data plotted in the plots is shown in the tables at the bottom of the page (data type):

1. Principal strains with fitted strain value (TPX).
2. Principal strain axes (TPX). The axes are also used to assign the eigenstrains to a particular direction (1,2 or 3). This is a difference from the original version of PASCal which just sorted by the magnitude of the strain. The directions are insensitive to inversion and are shown with the largest component positive.
3. Calculated derivative of the strain with parameter (PX).
4. Volume with fitted volume (TPX).
5. Parsed input values (TPX). These allow you to check the values have been read correctly in the event of anomalous results.

# Errors
PASCal does not use uncertainties in cell parameters or volumes in its fitting or strain calculations only using changing parameter (T/P/X). The uncertainties in T/P/X are used as weights for both linear and non-linear fitting but are not directly propagated through to the final parameter uncertainties. This usually provides a reasonable approximation as typically the inherent scatter of the strain is the primary source of uncertainty in the fitted parameter. In the case that the data has very small errors, this method will overestimate the magnitude of the errors. It can also prove inaccurate for small data sets.

It is possible if you have the magnitudes of the errors on your lattice parameters to propagate those errors through to get more accurate estimates, and [some software packages](#other-useful-software) (e.g. WINStrain) can give you errors on strains from errors on lattice parameters. However, to get very accurate estimates of errors it is also necessary to include the covariance of lattice parameters and for low symmetry systems, the propagation of errors in lattice parameter angles to strain errors is a problem with no unique solution. Additionally, unfortunately, the errors that are calculated from Rietveld fits or single crystal calculations often seem to underestimate the true errors in lattice parameters and so will underestimate the true error in your calculated values if used naively (see Haestier, J Appl. Cryst 2009, 42, 798, Herbstein, Acta Cryst B 2000, 56, 547; Taylor, R. & Kennard, O. Acta Cryst B 1986, 42, 112.)

In PASCal, linear fits use White’s heteroscedascity consistent error estimates and non-linear fits use the residuals to calculate the error estimates.

# Fitting
PASCal as an automated fitting tool cannot guarantee always finding the best fit and it is worth considering whether another [software package](#other-useful-software) to carry out these kind of calculation if they wish to examine this data more finely. There are some specific issues that users should consider, especially for high pressure data sets:

- PASCal does not use cell-parameter or volume errors in fitting or strain calculation, which can skew results. The contribution of volume errors to final errors is discussed in Angel (Reviews in Mineralogy and Geochemistry, Vol. 41, High-Temperature and High Pressure Crystal Chemistry 2000).

- Fitting to non-linear equations of state can be unstable, especially if the equation of state does not describe the behaviour well. Even where this instability occurs, it does not often seem to perturb the fitted parameters too badly. The errors, being derived from the derivatives of the parameters, are much more susceptible to instabilities and failure to converge. As the input errors (as currently implemented) are just weights, multiplying the errors by a constant can sometimes nudge the model towards convergence.

 - Ambient pressure (or critical pressure) data points can have outsize effects on the fitted parameters for both the Birch Murnaghan and empirical equations of state. Care should be taken with this point.
As different equations of state are used for the principal axis and volume fits, they should not be expected to produce self-consistent results. Other equations of state are of course in use and may be more appropriate for some materials.

# Strain calculation
PASCal calculates finite Lagrangian strains by default. Options to use finite and/or Lagrangian strains are provided. PASCal also uses a numerical approach that means the strains calculated are effectively averages of both direction and magnitude. It assigns the strain to the axes by comparing the eigenvectors to the first data point, however this approach can fail where there is significant eigenvector rotation or very noisy data. Fitting of the calculated strains outside of PASCal and manually assigning the strains to the appropriate axes is currently the best approach for these kinds of systems. The directions of the principal axes are taken, like the expansivities from the median data point.

# Offline Installation
PASCal can be also run offline using a browser as a GUI. This requires NumPy, Plotly and Flask. Full requirements (with versions) available in `requirements.txt`. Once the code is downloaded run the program using `flask run` in the folder containing `app.py`. The software has been designed to run as a web app, but please do submit any issues to the [GitHub Repository](./CONTRIBUTING.md).

# Issues and Feature Requests
If you find any bugs in the code, errors in the documentation or have any feature requests, please do [contribute]](./CONTRIBUTING.md) via the GitHub Repository.

# Other Useful Software
There are a wide range of other useful programs available:
- WINstrain and EOSfit (https://www.rossangel.com/). WINStrain provides a large number of different options for
calculating strains from lattice parameters, but is no longer supported. EOSfit is a powerful tool for  fitting equations of state (principally pressure).
- STRAIN (https://www.cryst.ehu.es/cryst/strain.html). The Bilbao Crystallographic Server can calculate strain calculations for a single pair of lattice parameters. 
- ELATE (https://progs.coudert.name/elate). A tool for analysing full elastic constant tensors.
If you know of any other useful software that should be added to the list or have any other questions,
please email matthew[dot]cliffe[at]chem[dot]ox[dot]ac[dot]uk