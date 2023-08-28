from dataclasses import dataclass
from typing import Union, List, Tuple, Any, Optional
from PASCal.options import Options, PASCalDataType
from PASCal.plotting import plot_strain, plot_volume
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
    """A container for the results of a PASCal run."""

    options: Options
    x: Union[Pressure, Temperature, Charge]
    x_errors: Union[Pressure, Temperature, Charge]
    diagonal_strain: Strain
    cell_volumes: Volume
    principal_components: np.ndarray
    indicatrix: Tuple
    norm_crax: np.ndarray
    median_x: int
    strain_fits: List[Any]
    volume_fits: List[Any]
    compressibility: Optional[np.ndarray]
    compressibility_errors: Optional[np.ndarray]
    warning: List[str]

    def plot_strain(self, return_json: bool = False, show_errors: bool = False):
        return plot_strain(
            self.x,
            self.x_errors,
            self.diagonal_strain,
            self.strain_fits,
            self.options.data_type,
            return_json=return_json,
            show_errors=show_errors,
        )

    def plot_volume(self, return_json: bool = False, show_errors: bool = False):
        return plot_volume(
            self.x,
            self.x_errors,
            self.cell_volumes,
            self.volume_fits,
            self.options.data_type,
            return_json=return_json,
            show_errors=show_errors,
        )


def _precheck_inputs(x, options) -> Tuple[Options, List[str]]:
    """Check that the raw data passed is compatible with the options, adjusting
    the options where possible.

    Returns:
        The adjusted options and a list of warnings.

    """
    if len(x) < 2:
        raise RuntimeError("Too few data points to perform fit: need at least 2")

    warning: List[str] = []

    if options.data_type == PASCalDataType.PRESSURE:
        if len(x) < 4:
            warning.append(
                "At least as many data points as parameters are needed for a fit to be carried out (e.g. 3 for 3rd order Birch-Murnaghan, 4 for empirical pressure fitting). "
                "As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."
            )
        if options.use_pc and options.pc_val:
            if np.amin(x) < options.pc_val:
                pc_val = np.min(x)
                warning.append(
                    "The critical pressure has to be smaller than the lower pressure data point. "
                    f"Critical pressure has been set to the minimum value: {pc_val} GPa."
                )
                options.pc_val = pc_val

    if options.data_type == PASCalDataType.ELECTROCHEMICAL:
        if len(x) - 2 < options.deg_poly_strain:
            deg_poly_strain = len(x) - 2
            warning.append(
                f"The maximum degree of the Chebyshev strain polynomial has been lowered from {options.deg_poly_strain} to {deg_poly_strain}. "
                "At least as many data points as parameters are needed for a fit to be carried out. "
                "As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."
            )
            options.deg_poly_strain = deg_poly_strain
        if len(x) - 2 < options.deg_poly_vol:
            deg_poly_vol = len(x) - 2
            warning.append(
                f"The maximum degree of the Chebyshev volume polynomial has been lowered from {options.deg_poly_vol} to {deg_poly_vol}. "
                "At least as many data points as parameters are needed for a fit to be carried out. "
                "As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."
            )
            options.deg_poly_vol = deg_poly_vol

    return options, warning


def fit(x, x_errors, unit_cells, options: Union[Options, dict]):
    if not isinstance(options, Options):
        options = Options.from_dict(options)

    options, warning = _precheck_inputs(x, options)
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

        bm_fits = ["bm2", "bm3", "bm3_pc"]
        for popts, pcovs, key in zip(bm_popts, bm_pcovs, bm_fits):
            volume_fits[key] = popts, pcovs

    elif options.data_type == PASCalDataType.ELECTROCHEMICAL:
        strain_fits["chebyshev"] = fit_chebyshev(
            diagonal_strain,
            x,
            options.deg_poly_strain,
        )
        volume_fits["chebyshev"] = fit_chebyshev(cell_volumes, x, options.deg_poly_vol)

    norm_crax = PASCal.utils.normalise_crys_axes(
        crys_prin_ax[median_x, :, :], principal_components
    )
    indicatrix = PASCal.utils.indicatrix(principal_components)

    return PASCalResults(
        options=options,
        x=x,
        x_errors=x_errors,
        principal_components=principal_components,
        median_x=median_x,
        diagonal_strain=diagonal_strain,
        cell_volumes=cell_volumes,
        strain_fits=strain_fits,
        volume_fits=volume_fits,
        indicatrix=indicatrix,
        norm_crax=norm_crax,
        warning=warning,
        compressibility=compressibility
        if options.data_type == PASCalDataType.PRESSURE
        else None,
        compressibility_errors=compressibility_errors
        if options.data_type == PASCalDataType.PRESSURE
        else None,
    )
