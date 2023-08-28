from typing import Dict, List, Union, Any
import json

import plotly
import plotly.graph_objs as go
import plotly.subplots
import numpy as np
import statsmodels.api as sm

from PASCal.options import PASCalDataType
from PASCal.constants import K_to_MK, PERCENT
from PASCal.utils import Pressure, Temperature, Charge, Strain, Volume

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
PLOT_INDICATRIX_LABELS: Dict[PASCalDataType, str] = {
    PASCalDataType.TEMPERATURE: "Expansivity (MK<sup>-1</sup>)",
    PASCalDataType.PRESSURE: "K (TPa<sup>-1</sup>)",
    PASCalDataType.ELECTROCHEMICAL: "Electrochemical strain charge derivative ([kAhg<sup>-1</sup>]<sup>-1</sup>)",
}
PLOT_X_LABELS = {
    PASCalDataType.TEMPERATURE: "Temperature (K)",
    PASCalDataType.PRESSURE: "Pressure (GPa)",
    PASCalDataType.ELECTROCHEMICAL: "Charge (mAhg<sup>-1</sup>)",
}


def plot_strain(
    x: Union[Pressure, Temperature, Charge],
    x_errors: Union[Pressure, Temperature, Charge],
    diagonal_strain: Strain,
    fit_results: Dict[str, Any],
    data_type: Union[PASCalDataType, str],
    show_errors: bool = False,
    return_json: bool = False,
) -> Union[go.Figure, str]:
    """Plot strain (%) against the fitted data in the 3 principal directions.

    Parameters:
        x: Array containing the control variable.
        x_errors: Array containing the errors in the control variable.
        diagonal_strain: The array of diagonal strains.
        fit_results: A list of statsmodel linear regression results for each direction.
        data_type: The PASCalDataType corresponding to the identity of the control variable.
        show_errors: Whether to show error bars on the plot.
        return_json: Whether to return the JSON dump of the plotly figure.

    Returns:
        JSON dump of the plotly figure if return_json, else return the plotly figure.

    """

    if not isinstance(data_type, PASCalDataType):
        data_type = PASCalDataType[data_type.upper()]

    if data_type == PASCalDataType.TEMPERATURE:
        # use linear fit results
        intercepts = [_.params[0] for _ in fit_results["linear"]]
        gradients = [_.params[1] for _ in fit_results["linear"]]
        ys = np.array([gradients[i] * x + intercepts[i] for i in range(3)])
        xs = x
    elif data_type == PASCalDataType.PRESSURE:
        # use empirical fit results
        popts, _, fns = fit_results["empirical"]
        xs = np.linspace(x[0], x[-1], num=1000)
        ys = np.array([fns(xs, *popts[i]) for i in range(3)])
    elif data_type == PASCalDataType.ELECTROCHEMICAL:
        # use chebyshev fits
        coeffs, _ = fit_results["chebyshev"]
        fns = [np.polynomial.chebyshev.Chebyshev(coeffs[i]) for i in range(3)]
        xs = x
        ys = np.array([fns[i](xs) for i in range(3)])

    figure = go.Figure()
    for i in range(3):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=diagonal_strain[:, i] * PERCENT,
                error_x=dict(type="data", array=x_errors, visible=show_errors),
                name=PLOT_STRAIN_LABEL[i],
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color=PLOT_PALETTE[i]),
            )
        )
        # line plot of fit plot
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys[i] * PERCENT,
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

    if return_json:
        json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return figure


def plot_volume(
    x: Union[Pressure, Temperature, Charge],
    x_errors: Union[Pressure, Temperature, Charge],
    cell_volumes: Volume,
    fit_result: Dict[str, Any],
    data_type: Union[PASCalDataType, str],
    show_errors: bool = False,
    return_json: bool = False,
) -> Union[go.Figure, str]:
    """Plot the cell volume fit.

    Parameters:
        x: Array containing the control variable.
        x_errors: Array containing the errors in the control variable.
        cell_volumes: array of cell volumes.
        fit_results: A statsmodel linear regression result for the cell volume fit.
        data_type: The PASCalDataType corresponding to the identity of the control variable.
        show_errors: Whether to show error bars on the plot.
        return_json: Whether or not to return the JSON dump of the figure.

    Returns:
        JSON dump of the plotly figure if return_json, else return the plotly figure.

    """

    if not isinstance(data_type, PASCalDataType):
        data_type = PASCalDataType[data_type.upper()]

    if data_type == PASCalDataType.TEMPERATURE:
        # use linear fit results
        intercepts = fit_result["linear"].params[0]
        gradient = fit_result["linear"].params[1]
        ys = gradient * x + intercept
        xs = x
    elif data_type == PASCalDataType.PRESSURE:
        # use empirical fit results
        popts, _, fns = fit_result["empirical"]
        xs = np.linspace(x[0], x[-1], num=1000)
        ys = fns(xs, *popts)
    elif data_type == PASCalDataType.ELECTROCHEMICAL:
        # use chebyshev fits
        coeffs, _ = fit_result["chebyshev"]
        fns = [np.polynomial.chebyshev.Chebyshev(coeffs[i]) for i in range(3)]
        xs = x
        ys = fns(xs)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=cell_volumes,
            error_x=dict(type="data", array=x_errors, visible=show_errors),
            name="V",
            mode="markers",
            marker_symbol="circle-open",
            marker=dict(color="Black"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
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

    if return_json:
        json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return figure


def plot_indicatrix(
    norm_crax,
    maxIn,
    R,
    X,
    Y,
    Z,
    data_type: Union[PASCalDataType, str],
    return_json: bool = False,
    plot_size: int = 800,
):
    if not isinstance(data_type, PASCalDataType):
        data_type = PASCalDataType[data_type.upper()]

    cbar_label = PLOT_INDICATRIX_LABELS[data_type]

    # draw crystallographic axes as arrows
    arrow_len = 1.6
    arrow_head = 0.2
    arrows = norm_crax * arrow_len

    figure = go.Figure()
    for i in range(3):
        figure.add_trace(
            go.Scatter3d(
                x=[0, arrows[i, 0]],
                y=[0, arrows[i, 1]],
                z=[0, arrows[i, 2]],
                mode="lines",
                line=dict(color="black", width=4),
                showlegend=False,
            )
        )

        figure.add_trace(
            go.Cone(
                x=[arrows[i, 0]],
                y=[arrows[i, 1]],
                z=[arrows[i, 2]],
                u=[arrows[i, 0] * arrow_head],
                v=[arrows[i, 1] * arrow_head],
                w=[arrows[i, 2] * arrow_head],
                anchor="cm",
                hoverinfo="skip",
                colorscale=[[0, "black"], [1, "black"]],
                showlegend=False,
                showscale=False,
            )
        )

    # Plot the indicatrix
    figure.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=R,
            cmax=maxIn,
            cmid=0,
            cmin=-maxIn,
            colorscale="rdbu_r",
            opacity=1.0,
            hovertemplate="alpha: %{surfacecolor:.1f}"
            + "<br>x: %{x:.1f}"
            + "<br>y: %{y:.1f}"
            + "<br>z: %{z:.1f}<extra></extra>",
            colorbar=dict(
                title=cbar_label,
                titleside="top",
                tickmode="array",
                ticks="outside",
            ),
        )
    )

    axis_scale = 2
    crax_label_pos = 1.1
    annotations = ["a", "b", "c"]
    axis_defaults = {
        "gridcolor": "grey",
        "showbackground": False,
        "range": [-maxIn * axis_scale, maxIn * axis_scale],
    }
    figure.update_layout(  # title='Indicatrix Plot',
        autosize=False,
        width=plot_size,
        height=plot_size,
        scene_aspectmode="cube",
        scene=dict(
            xaxis=axis_defaults,
            yaxis=axis_defaults,
            zaxis=axis_defaults,
            annotations=[
                dict(
                    showarrow=False,
                    x=norm_crax[i][0] * crax_label_pos * arrow_len,
                    y=norm_crax[i][1] * crax_label_pos * arrow_len,
                    z=norm_crax[i][2] * crax_label_pos * arrow_len,
                    text=annotations[i],
                    font=dict(color="black", size=15),
                )
                for i in range(3)
            ],
        ),
    )

    if return_json:
        return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return figure
