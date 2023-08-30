from typing import Tuple
import os
import json

from PASCal.constants import PERCENT
from PASCal.core import PASCalResults, fit
from PASCal.options import Options, PASCalDataType
import PASCal.utils

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
            Axes=["X<sub>1</sub>", "X<sub>2</sub>", "X<sub>3</sub>", "V"],
            PrinComp=np.round(results.principal_components, 4),
            MedianPrinAxCryst=PASCal.utils.round_array(
                results.median_principal_axis_crys, 4
            ),
            Vol=PASCal.utils.round_array(results.cell_volumes, 4),
            PrinAxCryst=PASCal.utils.round_array(results.principal_axis_crys, 4),
            TPx=results.x,
            DiagStrain=np.round(results.diagonal_strain * PERCENT, 4),
            TPxError=results.x_errors,
            Latt=results.unit_cells,
            **{
                k: PASCal.utils.round_array(results.named_coefficients[k], 4)
                for k in results.named_coefficients
            },
        )

    elif results.options.data_type == PASCalDataType.PRESSURE:
        return render_template(
            "pressure.html",
            config=json.dumps(PLOTLY_CONFIG),
            warning=results.warning,
            PlotStrainJSON=results.plot_strain(return_json=True),
            PlotVolumeJSON=results.plot_volume(return_json=True),
            PlotIndicJSON=results.plot_indicatrix(return_json=True),
            PlotDerivJSON=results.plot_compressibility(return_json=True),
            Axes=["X<sub>1</sub>", "X<sub>2</sub>", "X<sub>3</sub>", "V"],
            PrinComp=np.round(results.principal_components, 4),
            MedianPrinAxCryst=PASCal.utils.round_array(
                results.median_principal_axis_crys, 4
            ),
            PrinAxCryst=PASCal.utils.round_array(results.principal_axis_crys, 4),
            BMOrder=["2nd", "3rd"] + ["3rd with Pc"] if results.options.use_pc else [],
            TPxError=results.x_errors,
            u=results.median_x,
            UsePc=results.options.use_pc,
            Latt=results.unit_cells,
            K=PASCal.utils.round_array(results.compressibility, 4),
            KErr=PASCal.utils.round_array(results.compressibility_errors, 4),
            TPx=results.x,
            DiagStrain=np.round(results.diagonal_strain * PERCENT, 4),
            Vol=PASCal.utils.round_array(results.cell_volumes, 4),
            **{
                k: PASCal.utils.round_array(results.named_coefficients[k], 4)
                for k in results.named_coefficients
            },
        )

    elif results.options.data_type == PASCalDataType.ELECTROCHEMICAL:
        return render_template(
            "electrochem.html",
            config=json.dumps(PLOTLY_CONFIG),
            warning=results.warning,
            PlotStrainJSON=results.plot_strain(return_json=True),
            PlotVolumeJSON=results.plot_volume(return_json=True),
            PlotIndicJSON=results.plot_indicatrix(return_json=True),
            PlotDerivJSON=results.plot_charge_derivative(return_json=True),
            PlotResidualJSON=results.plot_residual(return_json=True),
            Axes=["X<sub>1</sub>", "X<sub>2</sub>", "X<sub>3</sub>", "V"],
            PrinComp=np.round(results.principal_components, 4),
            MedianPrinAxCryst=PASCal.utils.round_array(
                results.median_principal_axis_crys, 4
            ),
            PrinAxCryst=PASCal.utils.round_array(results.principal_axis_crys, 4),
            TPxError=results.x_errors,
            Latt=results.unit_cells,
            TPx=results.x,
            DiagStrain=np.round(results.diagonal_strain * PERCENT, 4),
            Vol=PASCal.utils.round_array(results.cell_volumes, 4),
            **{
                k: PASCal.utils.round_array(results.named_coefficients[k], 4)
                for k in results.named_coefficients
            },
        )


if __name__ == "__main__":
    app.run(debug=True)
