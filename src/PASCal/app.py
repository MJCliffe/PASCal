from typing import Tuple, List, Optional, Union
import json
import os

import PASCal._legacy
import PASCal.utils
from PASCal.fitting import (
    fit_linear_wls,
    fit_empirical_strain_pressure,
    fit_birch_murnaghan_volume_pressure,
    fit_chebyshev,
)
from PASCal.constants import PERCENT, K_to_MK, GPa_to_TPa, mAhg_to_kAhg
from PASCal.options import Options, PASCalDataType

from flask import Flask, render_template, request, send_from_directory

import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    print("Request for index page received")
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


def _parse_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses the string data provided in the web form into 3 arrays for
    x-variable, errors and unit cell parameters given x.

    Returns:
        A tuple of x, x_error, and unit_cell parameters.

    """
    raw_data = request.form.get("data")
    if not raw_data:
        raise RuntimeError("No data provided.")

    data = np.loadtxt(
        (line for line in raw_data.splitlines()),
    )

    x = data[:, 0]
    x_error = data[:, 1]
    unit_cells = data[:, 2:]

    return x, x_error, unit_cells


@app.route("/output", methods=["POST"])
def output():
    try:
        options = Options.from_form(request.form)
    except Exception as exc:
        raise RuntimeError(f"Could not parse options: {request.form}\nException: {exc}")

    raw_data = request.form.get("data")
    try:
        x, x_errors, unit_cells = _parse_data()
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse data: {request.form.get('data')}\nException: {exc}"
        )

    fit_results = fit(x, x_errors, unit_cells, options, raw_data)

    return _render_page(fit_results, options)


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

    ## Linear fitting of volume and lattice parameters
    strain_linear_fits = fit_linear_wls(diagonal_strain, x, x_errors)
    volume_linear_fits = fit_linear_wls(cell_volumes, x, x_errors)[0]

    if options.data_type == PASCalDataType.TEMPERATURE:
        principal_components = [
            strain_linear_fits[i].params[1] * K_to_MK for i in range(3)
        ]

    elif options.data_type == PASCalDataType.PRESSURE:
        # do empirical fits
        (
            strain_fits,
            strain_covs,
            empirical_stress_pressure,
        ) = fit_empirical_strain_pressure(
            diagonal_strain,
            x,
            x_errors,
            strain_linear_fits,
        )

        K = np.zeros_like(diagonal_strain)
        for i in range(3):
            K[i] = PASCal.utils.get_compressibility(x, strain_fits[i][1:]) * GPa_to_TPa
        principal_components = [K[i][median_x] for i in range(3)]

        bm_popts, bm_pcovs = fit_birch_murnaghan_volume_pressure(
            cell_volumes,
            x,
            x_errors,
            critical_pressure=options.pc_val if options.use_pc else None,
        )
    elif options.data_type == PASCalDataType.ELECTROCHEMICAL:
        strain_cheby_coeffs, strain_cheby_residuals = fit_chebyshev(
            diagonal_strain,
            x,
            options.deg_poly_strain,
        )
        volume_cheby_coeffs, volume_cheby_residuals = fit_chebyshev(
            cell_volumes, x, options.deg_poly_vol
        )

    ## return the data to the page ##return every plots with all the names then if else for each input in HTML
    indicatrix = PASCal.utils.indicatrix(principal_components)

    return (
        diagonal_strain,
        cell_volumes,
        strain_linear_fits,
        volume_linear_fits,
    )


def _render_page(fit_results, options: Options):
    if options.data_type == PASCalDataType.TEMPERATURE:
        response = render_template(
            "temperature.html",
            config=plotly_config,
            warning=warning,
            PlotStrainJSON=StrainJSON,
            PlotVolumeJSON=VolumeJSON,
            PlotIndicJSON=IndicatrixJSON,
            CoeffThermHeadings=CoeffThermHeadings,
            StrainHeadings=StrainHeadings,
            VolTempHeadings=VolTempHeadings,
            InputHeadings=InputHeadings,
            data=raw_data,
            Axes=Axes,
            PrinComp=np.round(PrinComp, 4),
            CalAlphaErr=PASCal._legacy.Round(CalAlphaErr * K_to_MK, 4),
            MedianPrinAxCryst=PASCal._legacy.Round(median_principal_axis_crys, 4),
            PrinAxCryst=PASCal._legacy.Round(principal_axis_crys, 4),
            TPx=x,
            DiagStrain=np.round(diagonal_strain * PERCENT, 4),
            XCal=PASCal._legacy.Round(XCal, 4),
            Vol=PASCal._legacy.Round(cell_volumes, 4),
            VolLin=PASCal._legacy.Round(VolLin, 4),
            VolCoef=PASCal._legacy.Round(VolCoef, 4),
            VolCoefErr=PASCal._legacy.Round(VolCoefErr, 4),
            TPxError=x_errors,
            Latt=unit_cells,
        )
        return response

    if options.data_type == PASCalDataType.PRESSURE:
        return render_template(
            "pressure.html",
            config=plotly_config,
            warning=warning,
            PlotStrainJSON=StrainJSON,
            PlotDerivJSON=DerivJSON,
            PlotVolumeJSON=VolumeJSON,
            PlotIndicJSON=IndicatrixJSON,
            KEmpHeadings=KEmpHeadings,
            CalEpsilon0=PASCal._legacy.Round(CalEpsilon0, 4),
            CalLambda=PASCal._legacy.Round(CalLambda, 4),
            CalPc=PASCal._legacy.Round(CalPc, 4),
            CalNu=PASCal._legacy.Round(CalNu, 4),
            StrainHeadings=StrainHeadings,
            InputHeadings=InputHeadings,
            data=raw_data,
            Axes=Axes,
            PrinComp=PASCal._legacy.Round(PrinComp, 4),
            KErr=PASCal._legacy.Round(KErr, 4),
            u=median_x,
            MedianPrinAxCryst=PASCal._legacy.Round(median_principal_axis_crys, 4),
            PrinAxCryst=PASCal._legacy.Round(principal_axis_crys, 4),
            BMCoeffHeadings=BMCoeffHeadings,
            BMOrder=BMOrder,
            B0=PASCal._legacy.Round(B0, 4),
            SigB0=PASCal._legacy.Round(SigB0, 4),
            V0=PASCal._legacy.Round(V0, 4),
            SigV0=PASCal._legacy.Round(SigV0, 4),
            BPrime=PASCal._legacy.Round(BPrime, 4),
            SigBPrime=SigBPrime,
            PcCoef=PASCal._legacy.Round(PcCoef, 4),
            KHeadings=KHeadings,
            K=PASCal._legacy.Round(K, 4),
            TPx=x,
            DiagStrain=PASCal._legacy.Round(diagonal_strain * PERCENT, 4),
            XCal=PASCal._legacy.Round(XCal, 4),
            VolPressHeadings=VolPressHeadings,
            Vol=PASCal._legacy.Round(cell_volumes, 4),
            VolCoef=PASCal._legacy.Round(VolCoef, 4),
            VolCoefErr=PASCal._legacy.Round(VolCoefErr, 4),
            CalPress=PASCal._legacy.Round(CalPress, 4),
            UsePc=str(options.use_pc),
            TPxError=x_errors,
            Latt=unit_cells,
        )

    if options.data_type == PASCalDataType.ELECTROCHEMICAL:
        return render_template(
            "electrochem.html",
            config=plotly_config,
            warning=warning,
            PlotStrainJSON=StrainJSON,
            PlotDerivJSON=DerivJSON,
            PlotVolumeJSON=VolumeJSON,
            PlotResidualJSON=ResidualJSON,
            PlotIndicJSON=IndicatrixJSON,
            QPrimeHeadings=QPrimeHeadings,
            data=raw_data,
            MedianPrinAxCryst=PASCal._legacy.Round(median_principal_axis_crys, 4),
            PrinAxCryst=PASCal._legacy.Round(principal_axis_crys, 4),
            TPx=x,
            Axes=Axes,
            PrinComp=PASCal._legacy.Round(PrinComp, 4),
            StrainHeadings=StrainHeadings,
            DiagStrain=PASCal._legacy.Round(diagonal_strain * PERCENT, 4),
            XCal=PASCal._legacy.Round(XCal, 4),
            DerHeadings=DerHeadings,
            Vol=PASCal._legacy.Round(cell_volumes, 4),
            Deriv=PASCal._legacy.Round(Deriv, 4),
            VolElecHeadings=VolElecHeadings,
            VolCheb=PASCal._legacy.Round(VolCheb, 4),
            VolCoef=PASCal._legacy.Round(VolCoef, 4),
            InputHeadings=InputHeadings,
            TPxError=x_errors,
            Latt=PASCal._legacy.Round(unit_cells, 4),
        )


if __name__ == "__main__":
    app.run(debug=True)
