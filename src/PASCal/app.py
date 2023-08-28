from typing import Tuple
import os
import json

import PASCal._legacy
from PASCal.constants import K_to_MK, PERCENT
from PASCal.core import PASCalResults, fit
from PASCal.options import Options, PASCalDataType

from flask import Flask, render_template, request, send_from_directory

import numpy as np

from PASCal.plotting import PLOTLY_CONFIG

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

    fit_results = fit(x, x_errors, unit_cells, options)

    return _render_results(fit_results)


def _render_results(results: PASCalResults) -> str:
    """Take the results of a PASCal fit and render them as HTML.

    Parameters:
        results: The results of a PASCal fit.

    Returns:
        The rendered HTML to serve.

    """

    if results.options.data_type == PASCalDataType.TEMPERATURE:
        return render_template(
            "temperature.html",
            config=json.dumps(PLOTLY_CONFIG),
            warning=results.warning,
            PlotStrainJSON=results.plot_strain(return_json=True),
            PlotVolumeJSON=results.plot_volume(return_json=True),
            PlotIndicJSON=results.plot_indicatrix(return_json=True),
            Axes=["X1", "X2", "X3", "V"],
            PrinComp=np.round(results.principal_components, 4),
            MedianPrinAxCryst=PASCal._legacy.Round(
                results.median_principal_axis_crys, 4
            ),
            Vol=PASCal.utils.round_array(results.cell_volumes, 4),
            PrinAxCryst=PASCal._legacy.Round(results.principal_axis_crys, 4),
            TPx=results.x,
            DiagStrain=np.round(results.diagonal_strain * PERCENT, 4),
            TPxError=results.x_errors,
            Latt=results.unit_cells,
            **{
                k: PASCal.utils.round_array(results.named_coefficients[k], 4)
                for k in results.named_coefficients
            },
        )

    if results.options.data_type == PASCalDataType.PRESSURE:
        pass
        # return render_template(
        #     "pressure.html",
        #     config=json.dumps(PLOTLY_CONFIG),
        #     warning=results.warning,
        #     PlotStrainJSON=results.plot_strain(return_json=True),
        #     PlotVolumeJSON=results.plot_volume(return_json=True),
        #     PlotIndicJSON=results.plot_indicatrix(return_json=True),
        #     PlotDerivJSON=DerivJSON,
        #     CalEpsilon0=PASCal._legacy.Round(CalEpsilon0, 4),
        #     CalLambda=PASCal._legacy.Round(CalLambda, 4),
        #     CalPc=PASCal._legacy.Round(CalPc, 4),
        #     CalNu=PASCal._legacy.Round(CalNu, 4),
        #     Axes=["X1", "X2", "X3", "V"],
        #     PrinComp=np.round(results.principal_components, 4),
        #     MedianPrinAxCryst=PASCal._legacy.Round(
        #         results.median_principal_axis_crys, 4
        #     ),
        #     PrinAxCryst=PASCal._legacy.Round(results.principal_axis_crys, 4),
        #     BMOrder=["2nd", "3rd"] + ["3rd with Pc"] if results.options.use_pc else [],
        #     B0=PASCal._legacy.Round(B0, 4),
        #     SigB0=PASCal._legacy.Round(SigB0, 4),
        #     V0=PASCal._legacy.Round(V0, 4),
        #     SigV0=PASCal._legacy.Round(SigV0, 4),
        #     BPrime=PASCal._legacy.Round(BPrime, 4),
        #     SigBPrime=SigBPrime,
        #     PcCoef=PASCal._legacy.Round(PcCoef, 4),
        #     CalPress=PASCal._legacy.Round(CalPress, 4),

        #     K=PASCal._legacy.Round(results.compressibility, 4),
        #     KErr=PASCal._legacy.Round(results.compressibility_errors, 4),
        # TPx=results.x,
        # DiagStrain=np.round(results.diagonal_strain * PERCENT, 4),
        # XCal=PASCal._legacy.Round(results.x, 4),
        # Vol=PASCal._legacy.Round(results.cell_volumes, 4),
        # VolCoef=PASCal._legacy.Round(
        #     results.volume_fits["linear"].params[1]
        #     / results.cell_volumes[0]
        #     * K_to_MK,
        #     4,
        # ),
    #         VolCoefErr=PASCal._legacy.Round(
    #             results.volume_fits["linear"].HC0_se[1]
    #             / results.cell_volumes[0]
    #             * K_to_MK,
    #             4,
    #         ),
    #         UsePc=results.options.use_pc,
    #         TPxError=results.x_errors,
    #         Latt=results.unit_cells,
    # )

    # if options.data_type == PASCalDataType.ELECTROCHEMICAL:
    #     return render_template(
    #         "electrochem.html",
    #         config=plotly_config,
    #         warning=warning,
    #         PlotStrainJSON=StrainJSON,
    #         PlotDerivJSON=DerivJSON,
    #         PlotVolumeJSON=VolumeJSON,
    #         PlotResidualJSON=ResidualJSON,
    #         PlotIndicJSON=IndicatrixJSON,
    #         QPrimeHeadings=QPrimeHeadings,
    #         data=raw_data,
    #         MedianPrinAxCryst=PASCal._legacy.Round(median_principal_axis_crys, 4),
    #         PrinAxCryst=PASCal._legacy.Round(principal_axis_crys, 4),
    #         TPx=x,
    #         Axes=Axes,
    #         PrinComp=PASCal._legacy.Round(PrinComp, 4),
    #         StrainHeadings=StrainHeadings,
    #         DiagStrain=PASCal._legacy.Round(diagonal_strain * PERCENT, 4),
    #         XCal=PASCal._legacy.Round(XCal, 4),
    #         DerHeadings=DerHeadings,
    #         Vol=PASCal._legacy.Round(cell_volumes, 4),
    #         Deriv=PASCal._legacy.Round(Deriv, 4),
    #         VolElecHeadings=VolElecHeadings,
    #         VolCheb=PASCal._legacy.Round(VolCheb, 4),
    #         VolCoef=PASCal._legacy.Round(VolCoef, 4),
    #         InputHeadings=InputHeadings,
    #         TPxError=x_errors,
    #         Latt=PASCal._legacy.Round(unit_cells, 4),
    #     )


if __name__ == "__main__":
    app.run(debug=True)
