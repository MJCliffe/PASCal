from typing import Dict, List
import json

import plotly
import plotly.graph_objs as go
import plotly.subplots
import numpy as np
import statsmodels.api as sm

from PASCal.options import PASCalDataType
from PASCal.constants import K_to_MK, PERCENT

PLOT_WIDTH: int = 500
PLOT_HEIGHT: int = 500
PLOT_MARGINS: Dict[str, float] = dict(t=50, b=50, r=50, l=50)
PLOT_PALETTE: List[str] = ["Red", "Green", "Blue"]
PLOT_STRAIN_LABEL: List[str] = ["ε<sub>1</sub>", "ε<sub>2</sub>", "ε<sub>3</sub>"]
PLOT_STRAIN_FIT_LABEL: List[str] = [
    "ε<sub>1,calc</sub>",
    "ε<sub>2,calc</sub>",
    "ε<sub>3,calc</sub>",
]

PLOT_X_LABELS = {
    PASCalDataType.TEMPERATURE: "Temperature (K)",
    PASCalDataType.PRESSURE: "Pressure (GPa)",
    PASCalDataType.ELECTROCHEMICAL: "Charge (mAhg<sup>-1</sup>)",
}


def plot_strain(
    x: np.ndarray,
    diagonal_strain: np.ndarray,
    fit_results: List[sm.regression.linear_model.RegressionResultsWrapper],
    data_type: PASCalDataType,
) -> str:
    """Plot strain (%) against the fitted data in the 3 principal directions.

    Parameters:
        x: Array containing the control variable.
        diagonal_strain: The array of diagonal strains.
        fit_results: A list of statsmodel linear regression results for each direction.
        data_type: The PASCalDataType corresponding to the identity of the control variable.

    Returns:
        JSON dump of the plotly figure.

    """

    if not len(fit_results) == 3:
        raise RuntimeError(
            "Expected 3 fitted models for strain plot, received {fit_results=}"
        )

    gradient = [_.params[1] for _ in fit_results]
    intercept = [_.params[0] for _ in fit_results]

    figure = go.Figure()
    for i in range(3):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=diagonal_strain[:, i] * PERCENT,
                name=PLOT_STRAIN_LABEL[i],
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color=PLOT_PALETTE[i]),
            )
        )
        # line plot of fit plot
        figure.add_trace(
            go.Scatter(
                x=x,
                y=(gradient[i] * x + cal_intercept[i]) * PERCENT,
                name=PLOT_STRAIN_FIT_LABEL[i],
                mode="lines",
                line=dict(color=PLOT_PALETTE[i]),
            )
        )

    figure.update_xaxes(
        title_text=PLOT_X_LABELS[data_type],
        mirror="ticks",
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    figure.update_yaxes(
        title_text="Relative change in <br> principal axis length (%)",
        mirror="ticks",
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    figure.add_hline(y=0, row=1, col=1)  # the horizontal line along the x axis

    figure.update_layout(
        autosize=False,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=PLOT_MARGINS,
        showlegend=True,
        hovermode="x unified",
        plot_bgcolor="white",
    )

    return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)


def plot_volume(
    x: np.ndarray,
    cell_volumes: np.ndarray,
    fit_result: sm.regression.linear_model.RegressionResultsWrapper,
    data_type: PASCalDataType,
) -> str:
    """Plot the cell volume fit.

    Parameters:
        x: Array containing the control variable.
        cell_volumes: array of cell volumes.
        fit_results: A statsmodel linear regression result for the cell volume fit.
        data_type: The PASCalDataType corresponding to the identity of the control variable.

    Returns:
        JSON dump of the plotly figure.

    """

    intercept = fit_result.params[0]
    gradient = fit_result.params[1]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=cell_volumes,
            name="V",
            mode="markers",
            marker_symbol="circle-open",
            marker=dict(color="Black"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=gradient * x + intercept,
            name="V<sub>lin</sub>",
            mode="lines",
            line=dict(color="Black"),
        )
    )

    figure.update_xaxes(
        title_text=PLOT_X_LABELS[data_type],
        mirror="ticks",
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    figure.update_yaxes(
        title_text="V (Å<sup>3</sup>)",
        mirror="ticks",
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    figure.update_layout(
        autosize=False,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=PLOT_MARGINS,
        showlegend=True,
        hovermode="x unified",
        plot_bgcolor="white",
    )

    return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
