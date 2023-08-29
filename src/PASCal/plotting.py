from typing import Dict, List, Union, Any
import json

import plotly
import plotly.graph_objs as go
import plotly.subplots
import numpy as np

from PASCal.options import PASCalDataType
from PASCal.constants import PERCENT, GPa_to_TPa, mAhg_to_kAhg
from PASCal.fitting import get_best_chebyshev_strain_fit, get_best_chebyshev_volume_fit
from PASCal.utils import (
    birch_murnaghan_2nd,
    birch_murnaghan_3rd,
    birch_murnaghan_3rd_pc,
    get_compressibility,
    Pressure,
    Temperature,
    Charge,
    Strain,
    Volume,
)

PLOT_WIDTH: int = 500
PLOT_HEIGHT: int = 500
PLOT_MARGINS: Dict[str, float] = dict(t=50, b=50, r=50, l=50)
PLOT_PALETTE: List[str] = ["Red", "Green", "Blue"]
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

PLOT_BM_LABELS = {
    birch_murnaghan_2nd: "V<sub>2nd BM</sub>",
    birch_murnaghan_3rd: "V<sub>3rd BM</sub>",
    birch_murnaghan_3rd_pc: "V<sub>3rd BM with P<sub>C</sub></sub>",
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",  # one of png, svg, jpeg, webp
        "scale": 5,  # change resolution of image
    },
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
        _, best_coeffs = get_best_chebyshev_strain_fit(fit_results["chebyshev"])
        fns = [np.polynomial.chebyshev.Chebyshev(best_coeffs[i]) for i in range(3)]
        xs = x
        ys = np.array([fns[i](xs) for i in range(3)])

    figure = go.Figure()
    for i in range(3):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=diagonal_strain[:, i] * PERCENT,
                error_x=dict(type="data", array=x_errors, visible=show_errors),
                name=f"ε<sub>{i}</sub>",
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
                name=f"ε<sub>{i},calc</sub>",
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
        return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return figure


def plot_volume(
    x: Union[Pressure, Temperature, Charge],
    x_errors: Union[Pressure, Temperature, Charge],
    cell_volumes: Volume,
    fit_result: Union[Dict[str, Any], List[Any]],
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
        intercept = fit_result["linear"].params[0]
        gradient = fit_result["linear"].params[1]
        ys = [gradient * x + intercept]
        xs = [x]
        labels = ["V<sub>lin</sub>"]
    elif data_type == PASCalDataType.PRESSURE:
        # use BM fit results
        xs = []
        ys = []
        labels = []
        for fn in fit_result:
            if fn == "linear":
                continue
            popts, _ = fit_result[fn]
            ys.append(np.linspace(cell_volumes[0], cell_volumes[-1], num=100))
            xs.append(fn(ys, *popts).reshape(-1))  # type: ignore
            labels.append(PLOT_BM_LABELS[fn])
    elif data_type == PASCalDataType.ELECTROCHEMICAL:
        # use chebyshev fits
        _, coeffs = get_best_chebyshev_volume_fit(fit_result["chebyshev"])
        fns = np.polynomial.chebyshev.Chebyshev(coeffs)
        xs = [x]
        ys = [fns(x)]
        labels = ["V<sub>cheb</sub>"]

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
    for ind, label in enumerate(labels):
        figure.add_trace(
            go.Scatter(
                x=xs[ind],
                y=ys[ind],
                name=label,
                mode="lines",
                line=dict(color="Black" if len(labels) == 1 else PLOT_PALETTE[ind]),
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
        return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return figure


def plot_compressibility(
    x: Pressure,
    compressibility: np.ndarray,
    compressibility_errors: np.ndarray,
    fit_results: Dict[str, Any],
    data_type: Union[PASCalDataType, str],
    return_json: bool = False,
):
    """Plot each component of the compressibility alongside
    the fitted values.

    Parameters:
        x: Array containing the control variable.
        compressibility: Array of compressibility values.
        compressibility_errors: Array of compressibility errors.
        fit_results: A set of `curve_fit` results for the compressibility fits in
            each direction, with key `empirical`.
        return_json: Whether or not to return the JSON dump of the figure.

    Returns:
        JSON dump of the plotly figure if return_json, else return the plotly figure.

    """
    figure = go.Figure()

    if not isinstance(data_type, PASCalDataType):
        data_type = PASCalDataType[data_type.upper()]
    if not data_type == PASCalDataType.PRESSURE:
        raise RuntimeError("compressibility plot only possible for pressure data")

    for i in range(3):
        k_label = f"K<sub>{i}</sub>"
        figure.add_trace(
            go.Scatter(
                name=k_label,
                x=x,
                y=compressibility[i],
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color=PLOT_PALETTE[i]),
            )
        )

        # TODO: fix
        # figure.add_trace(
        #     go.Scatter(
        #         x=np.concatenate([x, x[::-1]]),
        #         y=np.concatenate(
        #             [
        #                 compressibility[i] - compressibility_errors[i],
        #                 (compressibility[i] - compressibility_errors[i])[::-1],
        #             ]
        #         ),
        #         fill="toself",
        #         fillcolor=PLOT_PALETTE[i],
        #         line=dict(color=PLOT_PALETTE[i]),
        #         name=k_label,
        #         hoverinfo="skip",
        #         opacity=0.25,
        #     )
        # )

        xs = np.linspace(x[0], x[-1], num=200)

        # figure.add_trace(
        #     go.Scatter(
        #         x=xs,
        #         y=get_compressibility(
        #             xs,
        #             fit_results["empirical"][i][0][1],
        #             fit_results["empirical"][i][0][2],
        #             fit_results["empirical"][i][0][3],
        #         )
        #         * GPa_to_TPa,
        #         mode="lines",
        #         name=k_label,
        #         line=dict(color=PLOT_PALETTE[i]),
        #     )
        # )

    figure.update_xaxes(
        title_text="Pressure (GPa)",
        mirror="ticks",
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    figure.update_yaxes(
        title_text="Compressibility (TPa <sup>–1</sup>)",
        mirror="ticks",
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    figure.add_hline(y=0)  # the horizontal line along the x axis

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
        return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return figure


def plot_charge_derivative(
    x: Charge,
    strain: np.ndarray,
    fit_results: Dict[str, Any],
    data_type: Union[PASCalDataType, str],
    return_json: bool = False,
):
    """Plot each component of the charge derivative alongside
    the fitted values.

    Parameters:
        x: Array containing the control variable.
        strain: Array of strain values.
        fit_results: A set of `chebfit` results for the charge in
            each direction, with key `chebyshev`.
        return_json: Whether or not to return the JSON dump of the figure.

    Returns:
        JSON dump of the plotly figure if return_json, else return the plotly figure.

    """
    figure = go.Figure()

    if not isinstance(data_type, PASCalDataType):
        data_type = PASCalDataType[data_type.upper()]
    if not data_type == PASCalDataType.ELECTROCHEMICAL:
        raise RuntimeError(
            "charge derivative plot only possible for electrochemical data"
        )

    figure = go.Figure()

    _, best_coeffs = get_best_chebyshev_strain_fit(fit_results["chebyshev"])

    cheby_deriv = [
        np.polynomial.chebyshev.chebder(best_coeffs[i], m=1, scl=1, axis=0)
        for i in range(3)
    ]
    derivative = [
        mAhg_to_kAhg * np.polynomial.chebyshev.chebval(x, cheby_deriv[i])
        for i in range(3)
    ]

    xs = np.linspace(x[0], x[-1], num=300)

    for i in range(3):
        qp_label = f"q'<sub>{i}</sub>"

        figure.add_trace(
            go.Scatter(
                x=x,
                y=derivative[i],
                name=qp_label,
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color=PLOT_PALETTE[i]),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=xs,
                y=np.polynomial.chebyshev.chebval(xs, cheby_deriv[i]) * mAhg_to_kAhg,
                mode="lines",
                name=qp_label,
                line=dict(color=PLOT_PALETTE[i]),
            )
        )

    figure.add_hline(y=0)
    figure.update_xaxes(
        title_text="Cumulative capacity (mAhg<sup>-1</sup>)",
        mirror="ticks",
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    figure.update_yaxes(
        title_text="Charge-derivative of the electrochemical strain' (1/[kAhg<sup>-1</sup>])",
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
        return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    return figure


def plot_residual(
    strain_fit_results: Dict[str, Any],
    volume_fit_results: Dict[str, Any],
    data_type: Union[PASCalDataType, str],
    return_json: bool = False,
):
    """Plot the residual values of the Chebyshev fits for electrochemical data.

    Parameters:
        strain_fit_results: A set of `chebfit` results for the strain vs charge.
        volume_fit_results: A set of `chebfit` results for the volume vs charge.
        data_type: The type of data being plotted.
        return_json: Whether or not to return the JSON dump of the figure.

    Returns:
        JSON dump of the plotly figure if return_json, else return the plotly figure.

    """
    figure = go.Figure()

    if not isinstance(data_type, PASCalDataType):
        data_type = PASCalDataType[data_type.upper()]
    if not data_type == PASCalDataType.ELECTROCHEMICAL:
        raise RuntimeError("residual plot only possible for electrochemical data")

    vol_degrees = np.array(sorted(list(volume_fit_results["chebyshev"].keys())))
    vol_residuals = np.array(
        [volume_fit_results["chebyshev"][deg][1] for deg in vol_degrees]
    ).reshape(-1)

    figure.add_trace(
        go.Scatter(
            x=vol_degrees,
            y=vol_residuals,
            name="V",
            mode="lines+markers",
            line=dict(color="black"),
            marker=dict(color="black"),
        )
    )
    for i in range(3):
        degrees = sorted(list(strain_fit_results["chebyshev"].keys()))
        residuals = [strain_fit_results["chebyshev"][deg][1][i] for deg in degrees]
        figure.add_trace(
            go.Scatter(
                x=degrees,
                y=residuals,
                name=f"ε'<sub>{i}</sub>",
                mode="lines+markers",
                line=dict(color=PLOT_PALETTE[i]),
                marker=dict(color=PLOT_PALETTE[i]),
            )
        )

    figure.update_xaxes(
        title_text="Degree of Chebyshev polynomial",
        mirror="ticks",
        ticks="inside",
        showline=True,
        linecolor="black",
    )
    figure.update_yaxes(
        title_text="Sum of squared residual",
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
        return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
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
