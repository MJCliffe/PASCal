from typing import Tuple, List
import json
import os

import PASCal._legacy
import PASCal.utils
from PASCal.fitting import fit_linear_wls, fit_empirical, fit_chebyshev
from PASCal.constants import PERCENT, K_to_MK, GPa_to_TPa, mAhg_to_kAhg
from PASCal.options import Options, PASCalDataType
from PASCal.plotting import plot_strain, plot_volume

from flask import Flask, render_template, request, send_from_directory

import numpy as np
from scipy.optimize import curve_fit

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

    return fit(x, x_errors, unit_cells, options, raw_data)


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
        if len(x) - 2 < options.deg_poly_cap:
            deg_poly_cap = len(x) - 2
            warning.append(
                f"The maximum degree of the Chebyshev strain polynomial has been lowered from {options.deg_poly_cap} to {deg_poly_cap}. "
                "At least as many data points as parameters are needed for a fit to be carried out. "
                "As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."
            )
            options.deg_poly_cap = deg_poly_cap
        if len(x) - 2 < options.deg_poly_vol:
            deg_poly_vol = len(x) - 2
            warning.append(
                f"The maximum degree of the Chebyshev volume polynomial has been lowered from {options.deg_poly_vol} to {deg_poly_vol}. "
                "At least as many data points as parameters are needed for a fit to be carried out. "
                "As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."
            )
            options.deg_poly_vol = deg_poly_vol

    return options, warning


def fit(x, x_errors, unit_cells, options, raw_data):
    options, warning = _precheck_inputs(x, options)
    cell_volumes = PASCal._legacy.CellVol(unit_cells)  # volumes in (Angstrom^3)

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
    PrinAxCryst = np.transpose(
        np.matmul(orthonormed_cells, principal_axes[:, :, :]), axes=[0, 2, 1]
    )  # Eigenvector projected on crystallographic axes, UVW
    CrystPrinAx = np.linalg.inv(
        PrinAxCryst
    )  # Unit Cell in Principal axes coordinates, å
    PrinAxCryst = (
        PrinAxCryst.T / (np.sum(PrinAxCryst**2, axis=2) ** 0.5).T
    ).T  # normalised to make UVW near 1

    ### Ensures the largest component of each eigenvector is positive to make comparing easier
    MaxAxis = np.argmax(
        np.abs(PrinAxCryst), axis=2
    )  # find the largest value of each eigenvector
    I, J = np.indices(MaxAxis.shape)
    Mask = PrinAxCryst[I, J, MaxAxis] < 0
    PrinAxCryst[Mask, :] *= -1
    # transpositions to take advantage of broadcasting, not maths
    MedianPrinAxCryst = PrinAxCryst[median_x]

    ## Linear fitting of volume and lattice parameters

    if options.data_type == PASCalDataType.TEMPERATURE:
        strain_fits = PASCal.utils.fit_linear_wls(diagonal_strain, x, x_errors)
        volume_fits = PASCal.utils.fit_linear_wls(cell_volumes, x, x_errors)
    elif options.data_type == PASCalDataType.PRESSURE:
        # do BM fits
        strain_fits = PASCal.utils.fit_bm(diagonal_strain, x, x_errors)
    elif options.data_type == PASCALDataType.ELECTROCHEMICAL:
        # do chebyshev fits

    if options.data_type == PASCalDataType.PRESSURE:
        ### Unit conversion?
        CalEmPopt = np.zeros((3, 4))  # optimised empirical parameters
        CalEmPcov = np.zeros((3, 4, 4))  # the estimated covariance of CalEmPopt
        K = np.zeros((3, unit_cells.shape[0]))  # compressibilities TPa-1
        KErr = np.zeros((3, unit_cells.shape[0]))  # errors in K TPa-1

        VolCoef = -1 * VolGrad / cell_volumes[0] * GPa_to_TPa
        VolCoefErr = VolGradErr / cell_volumes[0] * GPa_to_TPa

        ### Bounds for the empirical fit
        EmpBounds = np.array(
            [
                [-np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, min(x), np.inf],
            ]
        )

        for i in range(3):
            InitParams = np.array([CalYInt[i], CalAlpha[i], min(x) - 0.001, 0.5])

            CalEmPopt[i], CalEmPcov[i] = curve_fit(
                PASCal._legacy.EmpEq,
                x,
                diagonal_strain[:, i],
                p0=InitParams,
                bounds=EmpBounds,
                maxfev=5000,
                sigma=x_errors,
            )
            XCal[i][:] = (
                PASCal._legacy.EmpEq(
                    x[:],
                    CalEmPopt[i][0],
                    CalEmPopt[i][1],
                    CalEmPopt[i][2],
                    CalEmPopt[i][3],
                )
                * PERCENT
            )  # strain %
            K[i][:] = (
                PASCal._legacy.Comp(
                    x[:], CalEmPopt[i][1], CalEmPopt[i][2], CalEmPopt[i][3]
                )
                * GPa_to_TPa
            )  # compressibilities (TPa^-1) so multiply by 1e3

            KErr[i][:] = (
                PASCal._legacy.CompErr(
                    CalEmPcov[i],
                    x[:],
                    CalEmPopt[i][1],
                    CalEmPopt[i][2],
                    CalEmPopt[i][3],
                )
                * GPa_to_TPa
            )  # errors in compressibilities (TPa^-1)

        CalEpsilon0 = np.array([CalEmPopt[0][0], CalEmPopt[1][0], CalEmPopt[2][0]])
        CalLambda = np.array([CalEmPopt[0][1], CalEmPopt[1][1], CalEmPopt[2][1]])
        CalPc = np.array([CalEmPopt[0][2], CalEmPopt[1][2], CalEmPopt[2][2]])
        CalNu = np.array([CalEmPopt[0][3], CalEmPopt[1][3], CalEmPopt[2][3]])
        PrinComp = np.array(
            [K[0][median_x], K[1][median_x], K[2][median_x]]
        )  # median compressibilities (TPa^-1) for indicatrix plot

        ### Volume fits
        PoptSecBM, PcovSecBM = curve_fit(
            PASCal._legacy.SecBM,
            cell_volumes,
            x,
            p0=np.array(
                [
                    cell_volumes[0],
                    -cell_volumes[0]
                    * (x[-1] - x[0])
                    / (cell_volumes[-1] - cell_volumes[0]),
                ]
            ),
            maxfev=5000,
            sigma=x_errors,
        )  # second-order Birch-Murnaghan fit
        SigV0SecBM, SigB0SecBM = np.sqrt(np.diag(PcovSecBM))

        IntBprime = (
            -cell_volumes[0] * (x[-1] - x[0]) / (cell_volumes[-1] - cell_volumes[0])
        ) / (
            x[-1] - x[0]
        )  # B prime=dB/dp  initial guess for the third-order Birch-Murnaghan fitting

        PoptThirdBM, PcovThirdBM = curve_fit(
            PASCal._legacy.ThirdBM,
            cell_volumes,
            x,
            p0=np.array([cell_volumes[0], PoptSecBM[1], IntBprime]),
            maxfev=5000,
            sigma=x_errors,
        )  # third-order Birch-Murnaghan fit
        SigV0ThirdBM, SigB0ThirdBM, SigBprimeThirdBM = np.sqrt(np.diag(PcovThirdBM))

        if options.use_pc:  ## if a critical pressure is USED
            PoptThirdBMPc, PcovThirdBMPc = curve_fit(
                PASCal._legacy.WrapperThirdBMPc(options.pc_val),
                cell_volumes,
                x,
                p0=np.array([PoptThirdBM[0], PoptThirdBM[1], PoptThirdBM[2]]),
                maxfev=5000,
                sigma=x_errors,
            )  # third order BM fit +Pc
            SigV0ThirdBMPc, SigB0ThirdBMPc, SigBprimeThirdBMPc = np.sqrt(
                np.diag(PcovThirdBMPc)
            )

        ### Birch-Murnaghan coefficients
        B0 = np.array([PoptSecBM[1], PoptThirdBM[1]])  # reference bulk modulus (GPa)
        SigB0 = np.array(
            [SigB0SecBM, SigB0ThirdBM]
        )  # standard error in reference bulk modulus (GPa)
        V0 = np.array(
            [PoptSecBM[0], PoptThirdBM[0]]
        )  # the reference volume V0 (Angstrom^3)
        SigV0 = np.array(
            [SigV0SecBM, SigV0ThirdBM]
        )  # pressure derivative of the bulk modulus (dimensionless)
        BPrime = np.array(
            [4, PoptThirdBM[2]]
        )  # pressure derivative of the bulk modulus (dimensionless)
        SigBPrime = [
            "n/a",
            PASCal._legacy.Round(SigBprimeThirdBM, 4),
        ]  #  the standard error in pressure derivative of the bulk modulus (dimensionless) - no applicable for 2nd order BM
        PcCoef = np.array([0, 0])
        if options.use_pc:  ## add in critical pressure values
            B0 = np.concatenate([B0, [PoptThirdBMPc[1]]])
            SigB0 = np.concatenate([SigB0, [SigB0ThirdBMPc]])
            V0 = np.concatenate([V0, [PoptThirdBMPc[0]]])
            SigV0 = np.concatenate([SigV0, [SigV0ThirdBMPc]])
            BPrime = np.concatenate([BPrime, [PoptThirdBMPc[2]]])
            SigBPrime.append(PASCal._legacy.Round(SigBprimeThirdBMPc, 4))
            PcCoef = np.concatenate([PcCoef, [options.pc_val]])

        ### Compute the pressure from all fits
        CalPress = np.zeros((3, unit_cells.shape[0]))
        CalPress[0][:] = (
            cell_volumes - VolYInt
        ) / VolGrad  # not the same as PASCal._legacy?
        CalPress[1][:] = PASCal._legacy.SecBM(
            cell_volumes[:], PoptSecBM[0], PoptSecBM[1]
        )
        CalPress[2][:] = PASCal._legacy.ThirdBM(
            cell_volumes[:], PoptThirdBM[0], PoptThirdBM[1], PoptThirdBM[2]
        )
        if options.use_pc:  ## if a critical pressure is USED
            PThirdBMPc = PASCal._legacy.ThirdBMPc(
                cell_volumes[:],
                PoptThirdBMPc[0],
                PoptThirdBMPc[1],
                PoptThirdBMPc[2],
                options.pc_val,
            )
            CalPress = np.vstack((CalPress, PThirdBMPc))

        ### Strain Plot
        FigStrain = go.Figure()
        for i in range(3):
            FigStrain.add_trace(
                go.Scatter(
                    name=StrainLabel[i],
                    x=x,
                    y=diagonal_strain[:, i] * PERCENT,
                    mode="markers",
                    marker_symbol="circle-open",
                    marker=dict(color=Colour[i]),
                )
            )  # strain
            FigStrain.add_trace(
                go.Scatter(
                    name=StrainFitLabel[i],
                    x=np.linspace(x[0], x[-1], num=1000),
                    y=PASCal._legacy.EmpEq(
                        np.linspace(x[0], x[-1], num=1000), *CalEmPopt[i]
                    )
                    * PERCENT,
                    mode="lines",
                    line=dict(color=Colour[i]),
                )
            )  # fit strain

        FigStrain.update_xaxes(
            title_text="Pressure (GPa)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigStrain.update_yaxes(
            title_text="Relative change in <br> principal axis length (%)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigStrain.add_hline(y=0)  # the horizontal line along the x axis

        FigStrain.update_layout(
            autosize=False,
            width=PlotWidth,
            height=PlotHeight,
            margin=PlotMargin,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
        )

        ### Compressibility Plot
        FigDeriv = go.Figure()
        for i in range(3):
            K_low = K[i] - KErr[i]
            K_high = K[i] + KErr[i]

            FigDeriv.add_trace(
                go.Scatter(
                    name=KLabel[i],
                    x=x,
                    y=K[i],
                    mode="markers",
                    marker_symbol="circle-open",
                    marker=dict(color=Colour[i]),
                )
            )

            FigDeriv.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([K_high, K_low[::-1]]),
                    fill="toself",
                    fillcolor=Colour[i],
                    line=dict(color=Colour[i]),
                    name=KLabel[i],
                    hoverinfo="skip",
                    opacity=0.25,
                )
            )

            FigDeriv.add_trace(
                go.Scatter(
                    name=KLabel[i],
                    x=np.linspace(x[0], x[-1], num=200),
                    y=PASCal._legacy.Comp(
                        np.linspace(x[0], x[-1], num=200),
                        CalEmPopt[i][1],
                        CalEmPopt[i][2],
                        CalEmPopt[i][3],
                    )
                    * GPa_to_TPa,  # compressibilities (TPa^-1) so multiply by 1e3,
                    mode="lines",
                    line=dict(color=Colour[i]),
                )
            )

        FigDeriv.update_xaxes(
            title_text="Pressure (GPa)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigDeriv.update_yaxes(
            title_text="Compressibility (TPa <sup>–1</sup>)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigDeriv.add_hline(y=0)  # the horizontal line along the x axis

        FigDeriv.update_layout(
            autosize=False,
            width=PlotWidth,
            height=PlotHeight,
            margin=PlotMargin,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
        )

        ### Volume plot
        FigVolume = go.Figure()

        FigVolume.add_trace(
            go.Scatter(
                name="V",
                x=x,
                y=cell_volumes,
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color="Black"),
            ),
        )  # data

        FigVolume.add_trace(
            go.Scatter(
                name="V<sub>2nd BM<sub>",
                x=PASCal._legacy.SecBM(
                    np.linspace(cell_volumes[0], cell_volumes[-1], num=100), *PoptSecBM
                ),
                y=np.linspace(cell_volumes[0], cell_volumes[-1], num=100),
                mode="lines",
                line=dict(color="Red"),
            )
        )  # BM 2nd

        FigVolume.add_trace(
            go.Scatter(
                name="V<sub>3rd BM<sub>",
                x=PASCal._legacy.ThirdBM(
                    np.linspace(cell_volumes[0], cell_volumes[-1], num=100),
                    *PoptThirdBM,
                ),
                y=np.linspace(cell_volumes[0], cell_volumes[-1], num=100),
                mode="lines",
                line=dict(color="Blue"),
            )
        )  # BM 3rd

        if options.use_pc:  ## add in critical pressure values
            FigVolume.add_trace(
                go.Scatter(
                    name="V<sub>3rd BM with P<sub>c</sub><sub>",
                    x=PASCal._legacy.ThirdBMPc(
                        np.linspace(cell_volumes[0], cell_volumes[-1], num=100),
                        *PoptThirdBMPc,
                        options.pc_val,
                    ),
                    y=np.linspace(cell_volumes[0], cell_volumes[-1], num=100),
                    mode="lines",
                    legendrank=16,
                    line=dict(color="Green"),
                )
            )  # BM3rd + PC

        FigVolume.update_xaxes(
            title_text="Pressure (GPa)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigVolume.update_yaxes(
            title_text="V (\u212B<sup>3</sup>)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )

        FigVolume.update_layout(
            autosize=False,
            width=PlotWidth,
            height=PlotHeight,
            margin=PlotMargin,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
        )

        StrainJSON = json.dumps(FigStrain, cls=plotly.utils.PlotlyJSONEncoder)
        DerivJSON = json.dumps(FigDeriv, cls=plotly.utils.PlotlyJSONEncoder)
        VolumeJSON = json.dumps(FigVolume, cls=plotly.utils.PlotlyJSONEncoder)

    if options.data_type == PASCalDataType.ELECTROCHEMICAL:
        ###
        TPxLabel = "q(mAhg^-1)"
        QPrimeLabel = ["q'<sub>1</sub>", "q'<sub>2</sub>", "q'<sub>3</sub>"]
        QPrimeHeadings = ["Axes", "q'(kAhg^-1)^-1", "a", "b", "c"]
        VolElecHeadings = [TPxLabel, "V(A^3)", "VCheb(A^3)"]
        DerHeadings = [TPxLabel, "q1'", "q2'", "q3'"]

        ### Chebyshev polynomial fits of strain
        CoefStrainList = (
            []
        )  # Chebyshev coefficients of all degrees of Chebyshev polynomials for each axis
        ResStrain = np.zeros(
            (3, options.deg_poly_cap)
        )  # Residuals of all degrees of Chebyshev polynomials for each axis
        ChebObj = []  # Chebyshev objects
        ChebStrainDeg = np.zeros(
            options.deg_poly_cap
        )  # degrees of Chebyshev polynomials
        Deriv = np.zeros(
            (3, unit_cells.shape[0])
        )  # derivatives of the Chebyshev polymoials
        ChebDer = np.zeros(
            (3, options.deg_poly_cap)
        )  # Chebyshev series coefficients of the derivative

        for i in range(0, 3):  # for every principal axis
            CoefAxis = []  # Chebyshev coefficients for each principal axis
            for j in range(1, options.deg_poly_cap + 1):  # for every degree
                ChebStrainDeg[j - 1] = int(
                    j
                )  # the degrees of Chebyshev polynomials for plotting the graph
                coef, FullList = np.polynomial.chebyshev.chebfit(
                    x, diagonal_strain[:, i], j, full=True
                )  # fitting
                CoefAxis.append(
                    coef
                )  # Chebyshev coefficients ordered from low to high for each degree
                ResStrain[i][j - 1] = FullList[0][0]  # include residual

            CoefStrainList.append(
                CoefAxis
            )  # append Chebyshev coefficients for each axis

            CoefStrainOp = CoefStrainList[i][
                np.argmin(ResStrain[i])
            ]  # Chebyshev coefficients that give the smallest residual for each axis
            XCal[i] = np.polynomial.chebyshev.chebval(
                x, CoefStrainOp
            )  # Chebyshev coefficients that give the smallest residual for each axis
            ChebObj.append(
                np.polynomial.chebyshev.Chebyshev(CoefStrainOp)
            )  # store Chebyshev objects for each axis
            ChebDer[i] = np.polynomial.chebyshev.chebder(
                CoefStrainOp, m=1, scl=1, axis=0
            )  # Chebyshev series coefficients of the derivative
            Deriv[i][:] = (
                np.polynomial.chebyshev.chebval(x, ChebDer[i]) * mAhg_to_kAhg
            )  # derivative at datapoints, now in muAhg^-1?

        PrinComp = np.array(
            [Deriv[0][median_x], Deriv[1][median_x], Deriv[2][median_x]]
        )  # median derivatives of the Chebyshev polynomial (1/muAhg^-1) for the indicatrix plot

        ### Chebyshev polynomial volume fit
        CoefVolList = (
            []
        )  # a list to store Chebyshev coefficients of all degrees of Chebyshev polynomials
        ResVol = np.zeros(
            options.deg_poly_vol
        )  # an array to store residuals of all degrees of Chebyshev polynomials
        ChebVolDeg = np.arange(1, options.deg_poly_vol + 1)

        for i in ChebVolDeg:  # for every degree(s) of Chebyshev polynomials
            coef, FullList = np.polynomial.chebyshev.chebfit(
                x, cell_volumes, i, full=True
            )  # fit
            ResVol[i - 1] = FullList[0][0]  # update residual
            CoefVolList.append(
                coef
            )  # append Chebyshev coefficients for each degree of Chebyshev polynomials

        CoefVolOp = CoefVolList[np.argmin(ResVol)]  # best cheb fit
        VolCheb = np.polynomial.chebyshev.chebval(
            x, CoefVolOp
        )  # calc volume from best Cheb fit
        ChebVolObj = np.polynomial.chebyshev.Chebyshev(CoefVolOp)
        VolDer = np.polynomial.chebyshev.chebder(CoefVolOp, m=1, scl=1, axis=0)
        VolCoef = np.polynomial.chebyshev.chebval(x, VolDer)[median_x] * mAhg_to_kAhg

        ### Strain Plot
        FigStrain = go.Figure()
        for i in range(3):  # adapt to plot every range
            FigStrain.add_trace(
                go.Scatter(
                    x=x,
                    y=diagonal_strain[:, i] * PERCENT,
                    name=StrainLabel[i],
                    mode="markers",
                    marker_symbol="circle-open",
                    marker=dict(color=Colour[i]),
                )
            )
            FigStrain.add_trace(
                go.Scatter(
                    x=x,
                    y=ChebObj[i](x) * PERCENT,
                    name=StrainFitLabel[i],
                    mode="lines",
                    line=dict(color=Colour[i]),
                )
            )

        FigStrain.update_xaxes(
            title_text="Cumulative capacity (mAhg<sup>-1</sup>)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigStrain.update_yaxes(
            title_text="Relative change in <br> principal axis length (%)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigStrain.add_hline(y=0)  # the horizontal line along the x axis

        FigStrain.update_layout(
            autosize=False,
            width=PlotWidth,
            height=PlotHeight,
            margin=PlotMargin,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
        )

        ### Deriv plot
        FigDeriv = go.Figure()
        for i in range(3):  # adapt to plot every range
            FigDeriv.add_trace(
                go.Scatter(
                    x=x,
                    y=Deriv[i],
                    name=QPrimeLabel[i],
                    legendgroup="5",
                    mode="markers",
                    marker_symbol="circle-open",
                    marker=dict(color=Colour[i]),
                )
            )
            FigDeriv.add_trace(
                go.Scatter(
                    x=np.linspace(x[0], x[-1], num=300),
                    y=np.polynomial.chebyshev.chebval(
                        np.linspace(x[0], x[-1], num=300), ChebDer[i]
                    )
                    * mAhg_to_kAhg,
                    name=QPrimeLabel[i],
                    mode="lines",
                    line=dict(color=Colour[i]),
                )
            )

        FigDeriv.add_hline(y=0)
        FigDeriv.update_xaxes(
            title_text="Cumulative capacity (mAhg<sup>-1</sup>)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigDeriv.update_yaxes(
            title_text="Charge-derivative of the electrochemical strain' (1/[kAhg<sup>-1</sup>])",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )

        FigDeriv.update_layout(
            autosize=False,
            width=PlotWidth,
            height=PlotHeight,
            margin=PlotMargin,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
        )

        ### Volume plot
        FigVolume = go.Figure()

        FigVolume.add_trace(
            go.Scatter(
                name="V",
                x=x,
                y=cell_volumes,
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color="Black"),
            ),
        )  # data

        FigVolume.add_trace(
            go.Scatter(
                x=x,
                y=ChebVolObj(x),
                name="V<sub>cheb</sub>",
                mode="lines",
                line=dict(color="Black"),
            )
        )
        FigVolume.update_xaxes(
            title_text="Cumulative capacity (mAhg<sup>-1</sup>)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigVolume.update_yaxes(
            title_text="V (\u212B<sup>3</sup>)",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )

        FigVolume.update_layout(
            autosize=False,
            width=PlotWidth,
            height=PlotHeight,
            margin=PlotMargin,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
        )

        ### Residual Plot
        FigResidual = go.Figure()
        for i in range(3):
            FigResidual.add_trace(
                go.Scatter(
                    x=ChebStrainDeg,
                    y=ResStrain[i],
                    name=StrainLabel[i],
                    mode="lines+markers",
                    line=dict(color=Colour[i]),
                    marker=dict(color=Colour[i]),
                )
            )

        FigResidual.add_trace(
            go.Scatter(
                x=ChebVolDeg,
                y=ResVol,
                name="V",
                mode="lines+markers",
                line=dict(color="Black"),
                marker=dict(color="Black"),
            )
        )

        FigResidual.update_xaxes(
            title_text="Degree of Chebyshev polynomial",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )
        FigResidual.update_yaxes(
            title_text="Sum of squared residual",
            mirror="ticks",
            ticks="inside",
            showline=True,
            linecolor="black",
        )

        FigResidual.update_layout(
            autosize=False,
            width=PlotWidth,
            height=PlotHeight,
            margin=PlotMargin,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
        )

        StrainJSON = json.dumps(FigStrain, cls=plotly.utils.PlotlyJSONEncoder)
        DerivJSON = json.dumps(FigDeriv, cls=plotly.utils.PlotlyJSONEncoder)
        VolumeJSON = json.dumps(FigVolume, cls=plotly.utils.PlotlyJSONEncoder)
        ResidualJSON = json.dumps(FigResidual, cls=plotly.utils.PlotlyJSONEncoder)

    ####
    ## Indicatrix 3D plot
    NormCrax = PASCal._legacy.NormCRAX(CrystPrinAx[median_x, :, :], PrinComp)
    maxIn, R, X, Y, Z = PASCal._legacy.Indicatrix(PrinComp)

    if options.data_type == PASCalDataType.TEMPERATURE:
        ColourBarTitle = "Expansivity (MK<sup>–1</sup>)"
    if options.data_type == PASCalDataType.ELECTROCHEMICAL:
        ColourBarTitle = "Electrochemical strain charge derivative ([kAhg<sup>–1</sup>]<sup>–1</sup>)"
    if options.data_type == PASCalDataType.PRESSURE:
        ColourBarTitle = "K (TPa<sup>–1</sup>)"

    FigIndic = go.Figure()

    for i in range(3):
        ### Plot the crystallographic axes
        ArrowLen = 1.6
        Arrow = NormCrax[i] * ArrowLen
        FigIndic.add_trace(
            go.Scatter3d(
                x=[0, Arrow[0]],
                y=[0, Arrow[1]],
                z=[0, Arrow[2]],
                mode="lines",
                line=dict(color="black", width=4),
                showlegend=False,
            )
        )

        ### Cone is used for the arrow for each axis
        ArrowHead = 0.2
        ArrowObject = go.Cone(
            x=[Arrow[0]],
            y=[Arrow[1]],
            z=[Arrow[2]],
            u=[Arrow[0] * ArrowHead],
            v=[Arrow[1] * ArrowHead],
            w=[Arrow[2] * ArrowHead],
            anchor="cm",
            hoverinfo="skip",
            colorscale=[[0, "black"], [1, "black"]],
            showlegend=False,
            showscale=False,
        )
        FigIndic.add_trace(ArrowObject)

    ### Plot the indicatrix
    FigIndic.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=R,
            cmax=maxIn,
            cmin=maxIn * -1,
            cmid=0,
            colorscale="rdbu_r",
            opacity=1.0,
            hovertemplate="alpha: %{surfacecolor:.1f}"
            + "<br>x: %{x:.1f}"
            + "<br>y: %{y:.1f}"
            + "<br>z: %{z:.1f}<extra></extra>",
            colorbar=dict(
                title=ColourBarTitle,
                titleside="top",
                tickmode="array",
                ticks="outside",
            ),
        )
    )

    ### Layout and annotations
    AxesScale = 2
    CraxLabelPos = 1.1
    GridColor = "grey"
    FigIndic.update_layout(
        # title='Indicatrix Plot',
        autosize=False,
        width=800,
        height=800,
        scene_aspectmode="cube",
        scene=dict(
            xaxis=dict(
                gridcolor=GridColor,
                # zerolinecolor='black',
                showbackground=False,
                range=[-1 * maxIn * AxesScale, maxIn * AxesScale],
            ),
            yaxis=dict(
                gridcolor=GridColor,
                # zerolinecolor='black',
                showbackground=False,
                range=[-1 * maxIn * AxesScale, maxIn * AxesScale],
            ),
            zaxis=dict(
                gridcolor=GridColor,
                # zerolinecolor='black',
                showbackground=False,
                range=[-1 * maxIn * AxesScale, maxIn * AxesScale],
            ),
            annotations=[
                dict(
                    showarrow=False,
                    x=NormCrax[0][0] * CraxLabelPos * ArrowLen,
                    y=NormCrax[0][1] * CraxLabelPos * ArrowLen,
                    z=NormCrax[0][2] * CraxLabelPos * ArrowLen,
                    text="a",
                    font=dict(color="black", size=15),
                ),
                dict(
                    showarrow=False,
                    x=NormCrax[1][0] * CraxLabelPos * ArrowLen,
                    y=NormCrax[1][1] * CraxLabelPos * ArrowLen,
                    z=NormCrax[1][2] * CraxLabelPos * ArrowLen,
                    text="b",
                    font=dict(color="black", size=15),
                ),
                dict(
                    showarrow=False,
                    x=NormCrax[2][0] * CraxLabelPos * ArrowLen,
                    y=NormCrax[2][1] * CraxLabelPos * ArrowLen,
                    z=NormCrax[2][2] * CraxLabelPos * ArrowLen,
                    text="c",
                    font=dict(color="black", size=15),
                ),
            ],
        ),
    )

    IndicatrixJSON = json.dumps(FigIndic, cls=plotly.utils.PlotlyJSONEncoder)

    ####
    ## Table labels
    # Axes = ["X<sub>1</sub>", "X<sub>2</sub>", "X<sub>3</sub>", "V"] #gets automatically escaped
    Axes = ["X1", "X2", "X3", "V"]
    StrainHeadings = [
        TPxLabel,
        "X<sub>1</sub>",
        "X<sub>2</sub>",
        "X<sub>3</sub>",
        "X<sub>1, calc</sub>",
        "X<sub>2, calc</sub>",
        "X<sub>3, calc</sub>",
    ]
    InputHeadings = [TPxLabel, "\u03C3T", "a", "b", "c", "ding", "beta", "gamma"]

    ####
    ## settings for plotly plots
    plotly_config = json.dumps(
        {
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",  # one of png, svg, jpeg, webp
                "scale": 5,  # change resolution of image
            },
        }
    )

    plot_strain(x, diagonal_strain, strain_fits, options.data_type)
    plot_volume(x, cell_volumes, volume_fits, options.data_type)

    ## return the data to the page ##return every plots with all the names then if else for each input in HTML

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
            MedianPrinAxCryst=PASCal._legacy.Round(MedianPrinAxCryst, 4),
            PrinAxCryst=PASCal._legacy.Round(PrinAxCryst, 4),
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
            MedianPrinAxCryst=PASCal._legacy.Round(MedianPrinAxCryst, 4),
            PrinAxCryst=PASCal._legacy.Round(PrinAxCryst, 4),
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
            MedianPrinAxCryst=PASCal._legacy.Round(MedianPrinAxCryst, 4),
            PrinAxCryst=PASCal._legacy.Round(PrinAxCryst, 4),
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
