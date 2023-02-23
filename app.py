from flask import Flask, render_template, request, send_from_directory
import plotly
import plotly.graph_objs as go
import plotly.subplots
import statsmodels.api as sm
import PASCal
import numpy as np
import json
import os

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


@app.route("/output", methods=["POST"])
def output():
    data = request.form.get("data")
    DataType = request.form.get(
        "DataType"
    )  # strings of 'Temperature' or 'Pressure' or 'Electrochemical'
    EulerianStrain = request.form.get("EulerianStrain")  # if false lagrangian
    FiniteStrain = request.form.get("Finite")  # true or false (Finite or infinitesimal)

    if DataType == "Pressure":
        UsePc = request.form.get(
            "UsePc"
        )  # if critical pressure is used (string, "true")
        if UsePc == "True":
            PcVal = float(request.form.get("PcVal"))  # critical pressure value
    if DataType == "Electrochemical":
        DegPolyCap = int(
            request.form.get("DegPolyCap")
        )  # degree of Chebyshev for strain
        DegPolyVol = int(
            request.form.get("DegPolyVol")
        )  # degree of Chebyshev for volume
    print("Request for output page received")

    ####
    # This section converts the raw string into a series of numpy arrays
    TPx = []  # Temperature or pressure or electrochemical data (K, TPa^-1, mAhg-1)
    TPxError = []  # error in input data (K, TPa^-1, mAhg-1)
    Latt = (
        []
    )  # lattice numpy array of lattice parameters, a b c alpha beta gamma (Angstrom, degrees)
    warning = ""

    for line in data.splitlines():
        line = line.strip()
        if not line.startswith("#"):  # ignore comments
            datum = np.fromstring(line, sep=" ")
            if np.shape(datum)[0] == 0:
                continue
            if np.shape(datum)[0] != 8:
                warning = (
                    "Wrong number of data points in this line: "
                    + str(datum)
                    + " has been ignored."
                )
                continue
            TPx.append(datum[0])
            TPxError.append(datum[1])
            Latt.append(datum[2:])

    TPx = np.array(TPx)
    TPxError = np.array(TPxError)
    Latt = np.stack(Latt)
    Vol = PASCal.CellVol(Latt)  # volumes in (Angstrom^3)

    if len(TPx) < 2:
        warning = "At least as many data points as parameters are needed for a fit to be carried out (i.e. 3 for 3rd order Birch-Murnaghan,4 for empirical pressure fitting). As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."

    if (len(TPx) < 4) and (DataType != "Temperature"):
        warning = "At least as many data points as parameters are needed for afit to be carried out (i.e. 3 for 3rd order Birch-Murnaghan,4 for empirical pressure fitting). As PASCal calculates errors from derivatives, more data points than parameters are needed for error estimates."

    if DataType == "Pressure":
        if UsePc == "True":
            if np.amin(TPx) < PcVal:
                PcVal = np.amin(TPx)
                warning = (
                    "The critical pressure has to be smaller than the lower pressure data point. Critical pressure has been set to the minimum value: "
                    + str(PcVal)
                )
    percent = 1e2  # conversions
    TPa = 1e3  # GPa to TPa
    MK = 1e6  # K to MK
    kAhg = 1e6  # mAhg-1 to kAhg-1

    u = int(np.ceil(len(TPx) / 2)) - 1  # median
    length = np.shape(Latt)[0]
    ####

    ### Strain calculations
    Orth = PASCal.Orthomat(Latt[:])  # cell in orthogonal axes
    if EulerianStrain:  # Strain calculation
        Strain = np.identity(3) - np.matmul(np.linalg.inv(Orth[0, :]), Orth[0:, :])
    else:
        Strain = np.matmul(np.linalg.inv(Orth[0:, :]), Orth[0, :]) - np.identity(3)
    if FiniteStrain:  # Symmetrising to remove shear
        Strain = (
            Strain
            + np.transpose(Strain, axes=[0, 2, 1])
            + np.matmul(Strain, np.transpose(Strain, axes=[0, 2, 1]))
        ) / 2
    else:
        Strain = (Strain + np.transpose(Strain, axes=[0, 2, 1])) / 2
    DiagStrain, PrinAx = np.linalg.eigh(
        Strain
    )  # Diagonalising to get eigenvectors and values, PrinAx is in orthogonal coordinates

    ### Axes matching
    for n in range(2, length):
        Match = np.dot(
            PrinAx[1, :], PrinAx[n,]
        )  # an array matching the axes against each other
        AxesOrder = np.zeros(
            (3), dtype=np.int8
        )  # a list of the axes needed to convert the eigenvalues and vectors into a consistent format
        for i in range(3):
            AxesOrder[i] = np.argmax(np.abs(Match[i, :]))
        for i in range(3):
            if np.count_nonzero(AxesOrder == i) != 1:
                AxesOrder = np.array(
                    (0, 1, 2), dtype=np.int8
                )  # if the auto match fails, set it back to normal
                print("Axes automatching failed for row: ", n, " TPx:", TPx[n])
        DiagStrain[n, [0, 1, 2]] = DiagStrain[n, AxesOrder]
        PrinAx[n, :, [0, 1, 2]] = PrinAx[n, :, AxesOrder]

    ### Calculating Eigenvectors and Cells in different coordinate systems
    PrinAxCryst = np.transpose(
        np.matmul(Orth, PrinAx[:, :, :]), axes=[0, 2, 1]
    )  # Eigenvector projected on crystallographic axes, UVW
    CrystPrinAx = np.linalg.inv(
        PrinAxCryst
    )  # Unit Cell in Principal axes coordinates, å
    PrinAxCryst = (
        PrinAxCryst.T / (np.sum(PrinAxCryst ** 2, axis=2) ** 0.5).T
    ).T  # normalised to make UVW near 1
    ### Ensures the largest component of each eigenvector is positive to make comparing easier
    MaxAxis = np.argmax(
        np.abs(PrinAxCryst), axis=2
    )  # find the largest value of each eigenvector
    I, J = np.indices(MaxAxis.shape)
    Mask = PrinAxCryst[I, J, MaxAxis] < 0
    PrinAxCryst[Mask, :] = PrinAxCryst[Mask, :] * -1
    # transpositions to take advantage of broadcasting, not maths
    MedianPrinAxCryst = PrinAxCryst[u]

    ####
    ## Linear fitting of volume and lattice parameters
    ## Linear strain fitting
    CalAlpha = np.zeros(3)
    CalAlphaErr = np.zeros(3)
    CalYInt = np.zeros(3)
    XCal = np.zeros((3, TPx.shape[0]))
    for i in range(3):
        X = sm.add_constant(TPx)

        StrainModel = sm.WLS(
            DiagStrain[:, i], X, weights=1 / TPxError
        )  # weighted least square
        StrainResults = StrainModel.fit()
        CalAlpha[i] = StrainResults.params[1]
        CalYInt[i] = StrainResults.params[0]
        CalAlphaErr[i] = StrainResults.HC0_se[1]
        XCal[i] = (CalAlpha[i] * TPx + CalYInt[i]) * percent  # for output

    ## Volume fitting
    X = sm.add_constant(TPx)
    VolModel = sm.WLS(Vol, X, weights=1 / TPxError)  # weighted least squares
    VolResults = VolModel.fit()
    VolGrad = VolResults.params[1]
    VolYInt = VolResults.params[0]
    VolGradErr = VolResults.HC0_se[1]
    VolLin = TPx * VolGrad + VolYInt

    # X = sm.add_constant(Vol)
    # VolModel = sm.WLS(TPx,X,weights=1/TPxError) #weighted least squares backward <- this doesn't really make sense
    # VolResults = VolModel.fit()
    # VolGrad = 1/VolResults.params[1]
    # VolYInt = -1*VolResults.params[0]/VolResults.params[1]
    # VolGradErr = VolResults.HC0_se[1]/VolResults.params[1]/VolResults.params[1]
    # VolLin = TPx*VolGrad+VolYInt

    ## Plotting
    Colour = ["Red", "Green", "Blue"]  # standard axes colours
    StrainLabel = ["\u03B5<sub>1</sub>", "\u03B5<sub>2</sub>", "\u03B5<sub>3</sub>"]
    StrainFitLabel = [
        "\u03B5<sub>1,calc</sub>",
        "\u03B5<sub>2,calc</sub>",
        "\u03B5<sub>3,calc</sub>",
    ]
    PlotWidth = 500
    PlotHeight = 500
    PlotMargin = dict(t=50, b=50, r=50, l=50)

    ## For each data type plot and do some additional fitting
    if DataType == "Temperature":
        ### headings for tables
        TPxLabel = "T(K)"
        CoeffThermHeadings = [
            "Axes",
            "\u03C3 (MK\u207b\u00B9)",
            "\u03C3\u03C3 (MK\N{SUPERSCRIPT ONE})",
            "a",
            "b",
            "c",
        ]
        VolTempHeadings = [TPxLabel, "V (A^3)", "VLin (A^3)"]

        ### unit conversions
        VolCoef = VolGrad / Vol[0] * MK
        VolCoefErr = VolGradErr / Vol[0] * MK
        PrinComp = CalAlpha * MK

        ### Strain Plot
        FigStrain = go.Figure()
        for i in range(3):
            ### Temperature (K) vs strain (percentage) graph along each axis
            FigStrain.add_trace(
                go.Scatter(
                    x=TPx,
                    y=DiagStrain[:, i] * percent,
                    name=StrainLabel[i],
                    mode="markers",
                    marker_symbol="circle-open",
                    marker=dict(color=Colour[i]),
                )
            )  # strain data
            FigStrain.add_trace(
                go.Scatter(
                    x=TPx,
                    y=(CalAlpha[i] * TPx + CalYInt[i]) * percent,
                    name=StrainFitLabel[i],
                    mode="lines",
                    line=dict(color=Colour[i]),
                )
            )  # strain linear fit

        FigStrain.update_xaxes(
            title_text="Temperature (K)",
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
        FigStrain.add_hline(y=0, row=1, col=1)  # the horizontal line along the x axis

        FigStrain.update_layout(
            autosize=False,
            width=PlotWidth,
            height=PlotHeight,
            margin=PlotMargin,
            showlegend=True,
            hovermode="x unified",
            plot_bgcolor="white",
        )

        ### Volume Plot
        FigVolume = go.Figure()
        FigVolume.add_trace(
            go.Scatter(
                x=TPx,
                y=Vol,
                name="V",
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color="Black"),
            )
        )
        FigVolume.add_trace(
            go.Scatter(
                x=TPx,
                y=VolLin,
                name="V<sub>lin</sub>",
                mode="lines",
                line=dict(color="Black"),
            )
        )  # linear fit for volume
        FigVolume.update_xaxes(
            title_text="Temperature (K)",
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
        VolumeJSON = json.dumps(FigVolume, cls=plotly.utils.PlotlyJSONEncoder)

    if DataType == "Pressure":

        ###Headings for tables
        TPxLabel = "P(GPa)"
        KEmpHeadings = [
            "Axes",
            "K(TPa-1)",
            "\u03C3K(TPa-1)",
            "a",
            "b",
            "c",
            "\u03B5",
            "\u03BB",
            "Pc",
            "\u03BD",
        ]
        KLabel = ["K<sub>1</sub>", "K<sub>2</sub>", "K<sub>3</sub>"]
        BMCoeffHeadings = [
            "",
            "B0 (GPa)",
            "\u03C3B0(GPa)",
            "V0(A^3)",
            "\u03C3V0(A^3)",
            "B'",
            "\u03C3B'",
            "Pc(GPa)",
        ]
        BMOrder = ["2nd", "3rd"]
        KHeadings = ["P", "K1", "K2", "K3", "\u03C3K1", "\u03C3K2", "\u03C3K3"]
        VolPressHeadings = [TPxLabel, "PLin", "PCalc,2nd", "P3rd", "V (A^3)"]

        if UsePc == "True":  ## if a critical pressure is USED
            BMOrder.append("3rd with Pc")
            VolPressHeadings = [
                TPxLabel,
                "PLin",
                "PCalc,2nd",
                "P3rd",
                "P3rd+Pc",
                "V (A^3)",
            ]

        ### Unit conversion?
        CalEmPopt = np.zeros((3, 4))  # optimised empirical parameters
        CalEmPcov = np.zeros((3, 4, 4))  # the estimated covariance of CalEmPopt
        K = np.zeros((3, Latt.shape[0]))  # compressibilities TPa-1
        KErr = np.zeros((3, Latt.shape[0]))  # errors in K TPa-1

        VolCoef = VolGrad / Vol[0] * TPa
        VolCoefErr = VolGradErr / Vol[0] * TPa

        ### Bounds for the empirical fit
        EmpBounds = np.array(
            [[-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, min(TPx), np.inf]]
        )

        for i in range(3):
            InitParams = np.array([CalYInt[i], CalAlpha[i], min(TPx) - 0.001, 0.5])
            CalEmPopt[i], CalEmPcov[i] = curve_fit(
                PASCal.EmpEq,
                TPx,
                DiagStrain[:, i],
                p0=InitParams,
                bounds=EmpBounds,
                maxfev=5000,
                sigma=TPxError,
            )
            XCal[i][:] = (
                PASCal.EmpEq(
                    TPx[:],
                    CalEmPopt[i][0],
                    CalEmPopt[i][1],
                    CalEmPopt[i][2],
                    CalEmPopt[i][3],
                )
                * percent
            )  # strain %
            K[i][:] = (
                PASCal.Comp(TPx[:], CalEmPopt[i][1], CalEmPopt[i][2], CalEmPopt[i][3])
                * TPa
            )  # compressibilities (TPa^-1) so multiply by 1e3
            KErr[i][:] = (
                PASCal.CompErr(
                    CalEmPcov[i],
                    TPx[:],
                    CalEmPopt[i][1],
                    CalEmPopt[i][2],
                    CalEmPopt[i][3],
                )
                * TPa
            )  # errors in compressibilities (TPa^-1)

        CalEpsilon0 = np.array([CalEmPopt[0][0], CalEmPopt[1][0], CalEmPopt[2][0]])
        CalLambda = np.array([CalEmPopt[0][1], CalEmPopt[1][1], CalEmPopt[2][1]])
        CalPc = np.array([CalEmPopt[0][2], CalEmPopt[1][1], CalEmPopt[2][2]])
        CalNu = np.array([CalEmPopt[0][3], CalEmPopt[1][3], CalEmPopt[2][3]])
        PrinComp = np.array(
            [K[0][u], K[1][u], K[2][u]]
        )  # median compressibilities (TPa^-1) for indicatrix plot

        ### Volume fits
        PoptSecBM, PcovSecBM = curve_fit(
            PASCal.SecBM,
            Vol,
            TPx,
            p0=np.array([Vol[0], -Vol[0] * (TPx[-1] - TPx[0]) / (Vol[-1] - Vol[0])]),
            maxfev=5000,
            sigma=TPxError,
        )  # second-order Birch-Murnaghan fit
        SigV0SecBM, SigB0SecBM = np.sqrt(np.diag(PcovSecBM))

        IntBprime = (-Vol[0] * (TPx[-1] - TPx[0]) / (Vol[-1] - Vol[0])) / (
            TPx[-1] - TPx[0]
        )  # B prime=dB/dp  initial guess for the third-order Birch-Murnaghan fitting

        PoptThirdBM, PcovThirdBM = curve_fit(
            PASCal.ThirdBM,
            Vol,
            TPx,
            p0=np.array([Vol[0], PoptSecBM[1], IntBprime]),
            maxfev=5000,
            sigma=TPxError,
        )  # third-order Birch-Murnaghan fit
        SigV0ThirdBM, SigB0ThirdBM, SigBprimeThirdBM = np.sqrt(np.diag(PcovThirdBM))

        if UsePc == "True":  ## if a critical pressure is USED
            PoptThirdBMPc, PcovThirdBMPc = curve_fit(
                PASCal.WrapperThirdBMPc(PcVal),
                Vol,
                TPx,
                p0=np.array([PoptThirdBM[0], PoptThirdBM[1], PoptThirdBM[2]]),
                maxfev=5000,
                sigma=TPxError,
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
            PASCal.Round(SigBprimeThirdBM, 4),
        ]  #  the standard error in pressure derivative of the bulk modulus (dimensionless) - no applicable for 2nd order BM
        PcCoef = np.array([0, 0])
        if UsePc == "True":  ## add in critical pressure values
            B0 = np.concatenate([B0, [PoptThirdBMPc[1]]])
            SigB0 = np.concatenate([SigB0, [SigB0ThirdBMPc]])
            V0 = np.concatenate([V0, [PoptThirdBMPc[0]]])
            SigV0 = np.concatenate([SigV0, [SigV0ThirdBMPc]])
            BPrime = np.concatenate([BPrime, [PoptThirdBMPc[2]]])
            SigBPrime.append(PASCal.Round(SigBprimeThirdBMPc, 4))
            PcCoef = np.concatenate([PcCoef, [PcVal]])

        ### Compute the pressure from all fits
        CalPress = np.zeros((3, Latt.shape[0]))
        CalPress[0][:] = (Vol - VolYInt) / VolGrad  # not the same as PASCal?
        CalPress[1][:] = PASCal.SecBM(Vol[:], PoptSecBM[0], PoptSecBM[1])
        CalPress[2][:] = PASCal.ThirdBM(
            Vol[:], PoptThirdBM[0], PoptThirdBM[1], PoptThirdBM[2]
        )
        if UsePc == "True":  ## if a critical pressure is USED
            PThirdBMPc = PASCal.ThirdBMPc(
                Vol[:], PoptThirdBMPc[0], PoptThirdBMPc[1], PoptThirdBMPc[2], PcVal
            )
            CalPress = np.vstack((CalPress, PThirdBMPc))

        ### Strain Plot
        FigStrain = go.Figure()
        for i in range(3):
            FigStrain.add_trace(
                go.Scatter(
                    name=StrainLabel[i],
                    x=TPx,
                    y=DiagStrain[:, i] * percent,
                    mode="markers",
                    marker_symbol="circle-open",
                    marker=dict(color=Colour[i]),
                )
            )  # strain
            FigStrain.add_trace(
                go.Scatter(
                    name=StrainFitLabel[i],
                    x=np.linspace(TPx[0], TPx[-1], num=1000),
                    y=PASCal.EmpEq(
                        np.linspace(TPx[0], TPx[-1], num=1000), *CalEmPopt[i]
                    )
                    * percent,
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
                    x=TPx,
                    y=K[i],
                    mode="markers",
                    marker_symbol="circle-open",
                    marker=dict(color=Colour[i]),
                )
            )

            FigDeriv.add_trace(
                go.Scatter(
                    x=np.concatenate([TPx, TPx[::-1]]),
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
                    x=np.linspace(TPx[0], TPx[-1], num=200),
                    y=PASCal.Comp(
                        np.linspace(TPx[0], TPx[-1], num=200),
                        CalEmPopt[i][1],
                        CalEmPopt[i][2],
                        CalEmPopt[i][3],
                    )
                    * TPa,  # compressibilities (TPa^-1) so multiply by 1e3,
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
                x=TPx,
                y=Vol,
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color="Black"),
            ),
        )  # data

        FigVolume.add_trace(
            go.Scatter(
                name="V<sub>2nd BM<sub>",
                x=PASCal.SecBM(np.linspace(Vol[0], Vol[-1], num=100), *PoptSecBM),
                y=np.linspace(Vol[0], Vol[-1], num=100),
                mode="lines",
                line=dict(color="Red"),
            )
        )  # BM 2nd

        FigVolume.add_trace(
            go.Scatter(
                name="V<sub>3rd BM<sub>",
                x=PASCal.ThirdBM(np.linspace(Vol[0], Vol[-1], num=100), *PoptThirdBM),
                y=np.linspace(Vol[0], Vol[-1], num=100),
                mode="lines",
                line=dict(color="Blue"),
            )
        )  # BM 3rd

        if UsePc == "True":  ## add in critical pressure values
            FigVolume.add_trace(
                go.Scatter(
                    name="V<sub>3rd BM with P<sub>c</sub><sub>",
                    x=PASCal.ThirdBMPc(
                        np.linspace(Vol[0], Vol[-1], num=100), *PoptThirdBMPc, PcVal
                    ),
                    y=np.linspace(Vol[0], Vol[-1], num=100),
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

    if DataType == "Electrochemical":
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
            (3, DegPolyCap)
        )  # Residuals of all degrees of Chebyshev polynomials for each axis
        ChebObj = []  # Chebyshev objects
        ChebStrainDeg = np.zeros(DegPolyCap)  # degrees of Chebyshev polynomials
        Deriv = np.zeros((3, Latt.shape[0]))  # derivatives of the Chebyshev polymoials
        ChebDer = np.zeros(
            (3, DegPolyCap)
        )  # Chebyshev series coefficients of the derivative

        for i in range(0, 3):  # for every principal axis
            CoefAxis = []  # Chebyshev coefficients for each principal axis
            for j in range(1, DegPolyCap + 1):  # for every degree
                ChebStrainDeg[j - 1] = int(
                    j
                )  # the degrees of Chebyshev polynomials for plotting the graph
                coef, FullList = np.polynomial.chebyshev.chebfit(
                    TPx, DiagStrain[:, i], j, full=True
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
                TPx, CoefStrainOp
            )  # Chebyshev coefficients that give the smallest residual for each axis
            ChebObj.append(
                np.polynomial.chebyshev.Chebyshev(CoefStrainOp)
            )  # store Chebyshev objects for each axis
            ChebDer[i] = np.polynomial.chebyshev.chebder(
                CoefStrainOp, m=1, scl=1, axis=0
            )  # Chebyshev series coefficients of the derivative
            Deriv[i][:] = (
                np.polynomial.chebyshev.chebval(TPx, ChebDer[i]) * kAhg
            )  # derivative at datapoints, now in muAhg^-1?

        PrinComp = np.array(
            [Deriv[0][u], Deriv[1][u], Deriv[2][u]]
        )  # median derivatives of the Chebyshev polynomial (1/muAhg^-1) for the indicatrix plot

        ### Chebyshev polynomial volume fit
        CoefVolList = (
            []
        )  # a list to store Chebyshev coefficients of all degrees of Chebyshev polynomials
        ResVol = np.zeros(
            DegPolyVol
        )  # an array to store residuals of all degrees of Chebyshev polynomials
        ChebVolDeg = np.arange(1, DegPolyVol + 1)

        for i in ChebVolDeg:  # for every degree(s) of Chebyshev polynomials
            coef, FullList = np.polynomial.chebyshev.chebfit(
                TPx, Vol, i, full=True
            )  # fit
            ResVol[i - 1] = FullList[0][0]  # update residual
            CoefVolList.append(
                coef
            )  # append Chebyshev coefficients for each degree of Chebyshev polynomials

        CoefVolOp = CoefVolList[np.argmin(ResVol)]  # best cheb fit
        VolCheb = np.polynomial.chebyshev.chebval(
            TPx, CoefVolOp
        )  # calc volume from best Cheb fit
        ChebVolObj = np.polynomial.chebyshev.Chebyshev(CoefVolOp)
        VolDer = np.polynomial.chebyshev.chebder(CoefVolOp, m=1, scl=1, axis=0)
        VolCoef = np.polynomial.chebyshev.chebval(TPx, VolDer)[u] * kAhg

        ### Strain Plot
        FigStrain = go.Figure()
        for i in range(3):  # adapt to plot every range
            FigStrain.add_trace(
                go.Scatter(
                    x=TPx,
                    y=DiagStrain[:, i] * percent,
                    name=StrainLabel[i],
                    mode="markers",
                    marker_symbol="circle-open",
                    marker=dict(color=Colour[i]),
                )
            )
            FigStrain.add_trace(
                go.Scatter(
                    x=TPx,
                    y=ChebObj[i](TPx) * percent,
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
                    x=TPx,
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
                    x=np.linspace(TPx[0], TPx[-1], num=300),
                    y=np.polynomial.chebyshev.chebval(
                        np.linspace(TPx[0], TPx[-1], num=300), ChebDer[i]
                    )
                    * kAhg,
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
                x=TPx,
                y=Vol,
                mode="markers",
                marker_symbol="circle-open",
                marker=dict(color="Black"),
            ),
        )  # data

        FigVolume.add_trace(
            go.Scatter(
                x=TPx,
                y=ChebVolObj(TPx),
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
    NormCrax = PASCal.NormCRAX(CrystPrinAx[u, :, :], PrinComp)
    maxIn, R, X, Y, Z = PASCal.Indicatrix(PrinComp)

    if DataType == "Temperature":
        ColourBarTitle = "Expansivity (MK<sup>–1</sup>)"
    if DataType == "Electrochemical":
        ColourBarTitle = "Electrochemical strain charge derivative ([kAhg<sup>–1</sup>]<sup>–1</sup>)"
    if DataType == "Pressure":
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
            colorscale="rdbu",
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
    config = json.dumps(
        {
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",  # one of png, svg, jpeg, webp
                "scale": 5,  # change resolution of image
            },
        }
    )

    ## return the data to the page ##return every plots with all the names then if else for each input in HTML
    if DataType == "Temperature":
        return render_template(
            "temperature.html",
            config=config,
            warning=warning,
            PlotStrainJSON=StrainJSON,
            PlotVolumeJSON=VolumeJSON,
            PlotIndicJSON=IndicatrixJSON,
            CoeffThermHeadings=CoeffThermHeadings,
            StrainHeadings=StrainHeadings,
            VolTempHeadings=VolTempHeadings,
            InputHeadings=InputHeadings,
            data=data,
            Axes=Axes,
            PrinComp=PASCal.Round(PrinComp, 4),
            CalAlphaErr=PASCal.Round(CalAlphaErr * MK, 4),
            MedianPrinAxCryst=PASCal.Round(MedianPrinAxCryst, 4),
            PrinAxCryst=PASCal.Round(PrinAxCryst, 4),
            TPx=TPx,
            DiagStrain=PASCal.Round(DiagStrain * percent, 4),
            XCal=PASCal.Round(XCal, 4),
            Vol=PASCal.Round(Vol, 4),
            VolLin=PASCal.Round(VolLin, 4),
            VolCoef=PASCal.Round(VolCoef, 4),
            VolCoefErr=PASCal.Round(VolCoefErr, 4),
            TPxError=TPxError,
            Latt=Latt,
        )

    if DataType == "Pressure":
        return render_template(
            "pressure.html",
            config=config,
            warning=warning,
            PlotStrainJSON=StrainJSON,
            PlotDerivJSON=DerivJSON,
            PlotVolumeJSON=VolumeJSON,
            PlotIndicJSON=IndicatrixJSON,
            KEmpHeadings=KEmpHeadings,
            CalEpsilon0=PASCal.Round(CalEpsilon0, 4),
            CalLambda=PASCal.Round(CalLambda, 4),
            CalPc=PASCal.Round(CalPc, 4),
            CalNu=PASCal.Round(CalNu, 4),
            StrainHeadings=StrainHeadings,
            InputHeadings=InputHeadings,
            data=data,
            Axes=Axes,
            PrinComp=PASCal.Round(PrinComp, 4),
            KErr=PASCal.Round(KErr, 4),
            u=u,
            MedianPrinAxCryst=PASCal.Round(MedianPrinAxCryst, 4),
            PrinAxCryst=PASCal.Round(PrinAxCryst, 4),
            BMCoeffHeadings=BMCoeffHeadings,
            BMOrder=BMOrder,
            B0=PASCal.Round(B0, 4),
            SigB0=PASCal.Round(SigB0, 4),
            V0=PASCal.Round(V0, 4),
            SigV0=PASCal.Round(SigV0, 4),
            BPrime=PASCal.Round(BPrime, 4),
            SigBPrime=SigBPrime,
            PcCoef=PASCal.Round(PcCoef, 4),
            KHeadings=KHeadings,
            K=PASCal.Round(K, 4),
            TPx=TPx,
            DiagStrain=PASCal.Round(DiagStrain * percent, 4),
            XCal=PASCal.Round(XCal, 4),
            VolPressHeadings=VolPressHeadings,
            Vol=PASCal.Round(Vol, 4),
            VolCoef=PASCal.Round(VolCoef, 4),
            VolCoefErr=PASCal.Round(VolCoefErr, 4),
            CalPress=PASCal.Round(CalPress, 4),
            UsePc=UsePc,
            TPxError=TPxError,
            Latt=Latt,
        )

    if DataType == "Electrochemical":
        return render_template(
            "electrochem.html",
            config=config,
            warning=warning,
            PlotStrainJSON=StrainJSON,
            PlotDerivJSON=DerivJSON,
            PlotVolumeJSON=VolumeJSON,
            PlotResidualJSON=ResidualJSON,
            PlotIndicJSON=IndicatrixJSON,
            QPrimeHeadings=QPrimeHeadings,
            data=data,
            MedianPrinAxCryst=PASCal.Round(MedianPrinAxCryst, 4),
            PrinAxCryst=PASCal.Round(PrinAxCryst, 4),
            TPx=TPx,
            Axes=Axes,
            PrinComp=np.round(PrinComp, 4),
            StrainHeadings=StrainHeadings,
            DiagStrain=np.round(DiagStrain * percent, 4),
            XCal=PASCal.Round(XCal, 4),
            DerHeadings=DerHeadings,
            Vol=PASCal.Round(Vol, 4),
            Deriv=PASCal.Round(Deriv, 4),
            VolElecHeadings=VolElecHeadings,
            VolCheb=PASCal.Round(VolCheb, 4),
            VolCoef=PASCal.Round(VolCoef, 4),
            InputHeadings=InputHeadings,
            TPxError=TPxError,
            Latt=PASCal.Round(Latt, 4),
        )


if __name__ == "__main__":
    app.run(debug=True)
