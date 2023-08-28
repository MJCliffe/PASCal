"""This module defines the core functionality of PASCal: the fit function and the results container."""

import plotly.graph_objs
from dataclasses import dataclass
from typing import Union, List, Tuple, Any, Optional, Dict
from PASCal.options import Options, PASCalDataType
from PASCal.plotting import plot_indicatrix, plot_strain, plot_volume
from PASCal.utils import Strain, Pressure, Volume, Charge, Temperature
from PASCal.constants import K_to_MK, GPa_to_TPa
from PASCal.fitting import (
    fit_linear_wls,
    fit_chebyshev,
    fit_birch_murnaghan_volume_pressure,
    fit_empirical_strain_pressure,
)
import PASCal.utils
import PASCal._legacy
import numpy as np


@dataclass
class PASCalResults:
    """A container for the results of a PASCal run, providing
    convenience wrappers to all data objects and plotting functions.

    """

    options: Options
    """The provided fit options."""

    x: Union[Pressure, Temperature, Charge]
    """The input control variable $X$, either pressure, temperature or charge."""
    x_errors: Union[Pressure, Temperature, Charge]
    """The input errors on the control variable $X$, either pressure, temperature or charge."""

    unit_cells: np.ndarray
    """The input unit cells."""

    diagonal_strain: Strain
    """The diagonalised strain."""

    cell_volumes: Volume
    """The computed cell volumes."""

    principal_components: np.ndarray
    """The eigenvalues of the strain."""

    indicatrix: Tuple
    """The indicatrix data used to construct the surface plot."""

    norm_crax: np.ndarray
    """The normalised crystal axes."""

    median_x: int
    """The index of the middle value of the control variable $X$."""

    median_principal_axis_crys: np.ndarray
    """The median principal axis of the strain."""

    principal_axis_crys: np.ndarray
    """The principal axes of the strain."""

    strain_fits: Dict[str, List[Any]]
    """A dictionary containing the strain fits. Under each key is a list of fit models for each principal axis,
    containing either a statsmodel fit result (temperature data), a tuple of `popt`, `pcov`, `callable` results from SciPy's `curve_fit` (pressure data),
    or an array of Chebyshev coefficients from NumPy's `chebfit` (electrochemical data).
    """
    volume_fits: Dict[str, Any]
    """A dictionary containing the volume fits. Under each key is a model for the cell volume,
    containing either a statsmodel fit result, a tuple of `popt`, `pcov`, `callable` results from SciPy's `curve_fit`,
    or an array of Chebyshev coefficients from NumPy's `chebfit`.
    """

    compressibility: Optional[np.ndarray]
    """The computed compressibility."""

    compressibility_errors: Optional[np.ndarray]
    """The computed compressibility errors."""

    warning: List[str]
    """Any warnings generated during the fit."""

    def plot_strain(
        self, return_json: bool = False, show_errors: bool = False
    ) -> Union[str, plotly.graph_objs.Figure]:
        """Plots the strain fits."""
        return plot_strain(
            self.x,
            self.x_errors,
            self.diagonal_strain,
            self.strain_fits,
            self.options.data_type,
            return_json=return_json,
            show_errors=show_errors,
        )

    def plot_volume(
        self, return_json: bool = False, show_errors: bool = False
    ) -> Union[str, plotly.graph_objs.Figure]:
        return plot_volume(
            self.x,
            self.x_errors,
            self.cell_volumes,
            self.volume_fits,
            self.options.data_type,
            return_json=return_json,
            show_errors=show_errors,
        )

    def plot_indicatrix(
        self, return_json: bool = False
    ) -> Union[str, plotly.graph_objs.Figure]:
        return plot_indicatrix(
            self.norm_crax,
            *self.indicatrix,
            self.options.data_type,
            return_json=return_json,
        )


def fit(x, x_errors, unit_cells, options: Union[Options, dict]) -> PASCalResults:
    """Perform the PASCal fits for the given data.

    For temperature data, linear models are fitted for strain vs temperature and volume
    vs temperature.
    For pressure data, an empirical fit is made for strain vs pressure, and
    Birch-Murnaghan fits are made for volume vs pressure.
    For electrochemical data, Chebyshev polynomials are fitted for strain vs state
    of charge and volume vs state of charge.

    Parameters:
        x: The independent variable (pressure, temperature or charge).
        x_errors: The errors on the independent variable.
        unit_cells: The unit cell volumes.
        options: The options for the fit.

    Returns:
        A PASCalResults object containing the results of the fit.

    """
    if not isinstance(options, Options):
        options = Options.from_dict(options)

    warning = options.precheck_inputs(x)
    print(f"Performing fit with {options=}")
    cell_volumes = PASCal._legacy.CellVol(unit_cells)

    orthonormed_cells = PASCal._legacy.Orthomat(unit_cells)  # cell in orthogonal axes

    strain = PASCal.utils.calculate_strain(
        orthonormed_cells,
        mode="eulerian" if options.eulerian_strain else "lagrangian",
        finite=options.finite_strain,
    )

    # Diagonalising to get eigenvectors and values, principal_axes are in orthogonal coordinates
    diagonal_strain, principal_axes = np.linalg.eigh(strain)

    median_x = int(np.ceil(len(x) / 2)) - 1  # median

    ### Axes matching
    principal_axes, diagonal_strain = PASCal.utils.match_axes(
        principal_axes, unit_cells, diagonal_strain
    )

    ### Calculating Eigenvectors and Cells in different coordinate systems
    (
        median_principal_axis_crys,
        principal_axis_crys,
        crys_prin_ax,
    ) = PASCal.utils.rotate_to_principal_axes(
        orthonormed_cells, principal_axes, median_x
    )

    strain_fits = {}
    volume_fits = {}

    strain_fits["linear"] = fit_linear_wls(diagonal_strain, x, x_errors)
    volume_fits["linear"] = fit_linear_wls(cell_volumes, x, x_errors)[0]

    if options.data_type == PASCalDataType.TEMPERATURE:
        principal_components = [
            strain_fits["linear"][i].params[1] * K_to_MK for i in range(3)
        ]

    elif options.data_type == PASCalDataType.PRESSURE:
        # do empirical fits
        strain_fits["empirical"] = fit_empirical_strain_pressure(
            diagonal_strain,
            x,
            x_errors,
            strain_fits["linear"],
        )

        compressibility = np.zeros((3, len(x)))
        for i in range(3):
            empirical_popts = strain_fits["empirical"][0][i][1:]
            compressibility[i][:] = (
                PASCal.utils.get_compressibility(x, *empirical_popts) * GPa_to_TPa
            )
        compressibility_errors = []
        principal_components = [compressibility[i][median_x] for i in range(3)]

        bm_popts, bm_pcovs = fit_birch_murnaghan_volume_pressure(
            cell_volumes,
            x,
            x_errors,
            critical_pressure=options.pc_val if options.use_pc else None,
        )

        for k in bm_popts:
            volume_fits[k] = bm_popts[k], bm_pcovs[k]

    elif options.data_type == PASCalDataType.ELECTROCHEMICAL:
        strain_fits["chebyshev"] = fit_chebyshev(
            diagonal_strain,
            x,
            options.deg_poly_strain,
        )
        volume_fits["chebyshev"] = fit_chebyshev(cell_volumes, x, options.deg_poly_vol)

        cheb_derivative = np.zeros((3, len(x)))
        # for i in range(3):
        # cheb_derivative[i] = np.polynomial.chebyshev.chebder(
        #     strain_fits["chebyshev"][i][0], m=1, scl=1, axis=0
        # )

        principal_components = [cheb_derivative[i][median_x] for i in range(3)]

    norm_crax = PASCal.utils.normalise_crys_axes(
        crys_prin_ax[median_x, :, :], principal_components
    )
    indicatrix = PASCal.utils.indicatrix(principal_components)

    return PASCalResults(
        options=options,
        x=x,
        x_errors=x_errors,
        unit_cells=unit_cells,
        principal_components=principal_components,
        median_x=median_x,
        diagonal_strain=diagonal_strain,
        cell_volumes=cell_volumes,
        strain_fits=strain_fits,
        volume_fits=volume_fits,
        indicatrix=indicatrix,
        norm_crax=norm_crax,
        median_principal_axis_crys=median_principal_axis_crys,
        principal_axis_crys=principal_axis_crys,
        warning=warning,
        compressibility=compressibility
        if options.data_type == PASCalDataType.PRESSURE
        else None,
        compressibility_errors=compressibility_errors
        if options.data_type == PASCalDataType.PRESSURE
        else None,
    )
