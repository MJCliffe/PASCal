import numpy as np


def Round(var, dec):
    # Rounding the number to desired decimal places
    # round() is more accurate at rounding float numbers than np.round()
    # NewVar = np.zeros((var.shape[0], var.shape[1]))
    if var.ndim == 1:
        for i in range(var.shape[0]):
            var[i] = round(var[i], dec)
    if var.ndim == 2:
        for i in range(var.shape[0]):
            for j in range(var.shape[1]):
                var[i][j] = round(var[i][j], dec)
    if var.ndim == 3:
        for i in range(var.shape[0]):
            for j in range(var.shape[1]):
                for k in range(var.shape[2]):
                    var[i][j][k] = round(var[i][j][k], dec)
    return var


def Orthomat(Latt):
    # Compute the corresponding change-of-basis transformation (square matrix M) in E = M x A
    # Latt: a b c alpha beta gamma (Angstrom, degrees)
    orth = np.zeros((np.shape(Latt)[0], 3, 3))
    alpha = Latt[:, 3] * (np.pi / 180)
    beta = Latt[:, 4] * (np.pi / 180)
    gamma = Latt[:, 5] * (np.pi / 180)
    gaS = np.arccos(
        (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))
    )
    orth[:, 0, 0] = 1 / (Latt[:, 0] * np.sin(beta) * np.sin(gaS))
    orth[:, 1, 0] = np.cos(gaS) / (Latt[:, 1] * np.sin(alpha) * np.sin(gaS))
    orth[:, 2, 0] = (
        np.cos(alpha) * np.cos(gaS) / np.sin(alpha) + np.cos(beta) / np.sin(beta)
    ) / (-1 * Latt[:, 2] * np.sin(gaS))
    orth[:, 1, 1] = 1 / (Latt[:, 1] * np.sin(alpha))
    orth[:, 2, 1] = -1 * np.cos(alpha) / (Latt[:, 2] * np.sin(alpha))
    orth[:, 2, 2] = 1 / Latt[:, 2]
    return orth
    # m11 = 1/(Latt[0]*np.sin(beta)*np.sin(gaS))
    # m21 = np.cos(gaS)/(Latt[1]*np.sin(alpha)*np.sin(gaS))
    # m22 = 1/(Latt[1]*np.sin(alpha))
    # m31 = (np.cos(alpha)*np.cos(gaS)/np.sin(alpha)+np.cos(beta)/np.sin(beta))/(-1*Latt[2]*np.sin(gaS))
    # m32 = -1*np.cos(alpha)/(Latt[2]*np.sin(alpha))
    # m33 = 1/Latt[2]
    # return np.array([[m11, 0, 0], [m21, m22, 0], [m31, m32, m33]])


def CellVol(LattPam):
    # Calculate the unit-cell volume
    # Lattice parameters a b c al be ga at the median point  (Angstrom, degrees)
    vol = np.zeros((LattPam.shape[0]))
    for i in range(0, LattPam.shape[0]):
        Latt = LattPam[i]
        vol[i] = (
            Latt[0]
            * Latt[1]
            * Latt[2]
            * (
                (1 - np.cos(Latt[3] * (np.pi / 180)) ** 2)
                - (np.cos(Latt[4] * (np.pi / 180)) ** 2)
                - (np.cos(Latt[5] * (np.pi / 180)) ** 2)
                + (
                    2
                    * np.cos(Latt[3] * (np.pi / 180))
                    * np.cos(Latt[4] * (np.pi / 180))
                    * np.cos(Latt[5] * (np.pi / 180))
                )
            )
            ** (0.5)
        )
    return vol


def EmpEq(TP, Epsilon0, lambdaP, Pc, Nu):
    # Empirical fit for pressure input data
    # TP: pressure data points, numpy array
    # Epsilon0: strain at critical pressure
    # lambdaP: compressibility (GPa^-nu)
    # Pc: critical pressure (GPa)
    # Nu: rate of softening 0.5
    return Epsilon0 + (lambdaP * ((TP - Pc) ** Nu))


def Comp(TP, lambdaP, Pc, Nu):
    # Calculate the compressibility from the derivative -de/dp
    return -lambdaP * Nu * ((TP - Pc) ** (Nu - 1))


def CompErr(Pcov, TP, lambdaP, Pc, Nu):
    # Calculate errors in compressibilities
    # Pcov: the estimated covariance of optimal values of the empirical parameters

    Jac = np.zeros((4, TP.shape[0]))  # jacobian matrix
    KErr = np.zeros(TP.shape[0])
    for n in range(0, len(TP)):
        Jac[0][n] = 0
        Jac[1][n] = ((TP[n] - Pc) ** (Nu - 1)) * Nu
        Jac[2][n] = -1 * lambdaP * Nu * (Nu - 1) * (TP[n] - Pc) ** (Nu - 2)
        Jac[3][n] = (Nu * np.log(TP[n] - Pc) + 1) * ((TP[n] - Pc) ** (Nu - 1)) * lambdaP
        KErrPoint = 0
        for j in range(0, 4):
            for i in range(0, 4):
                KErrPoint = KErrPoint + Jac[j][n] * Jac[i][n] * Pcov[j][i]
        KErr[n] = KErrPoint ** 0.5
    return KErr


def Eta(V, V0):
    # Defining the parameter to be used in Birch-Murnaghan equations of state
    # V: unit-cell volume at a pressure point
    # V0: the zero pressure unit-cell volume
    return np.abs(V0 / V) ** (1 / 3)


def SecBM(V, V0, B):
    # The second-order Birch-Murnaghan fit corresponding the equation of state
    # V: unit-cell volume at a pressure point (Angstrom^3)
    # V0: the zero pressure unit-cell volume (Angstrom^3)
    # B: Bulk modulus (GPa)
    return (3 / 2) * B * (Eta(V, V0) ** 7 - Eta(V, V0) ** 5)


def ThirdBM(V, V0, B0, Bprime):
    # The third-order Birch-Murnaghan fit corresponding the equation of state
    # V: unit-cell volume at a pressure point (Angstrom^3)
    # V0: the zero pressure unit-cell volume (Angstrom^3)
    # B0: Bulk modulus at zero pressure (GPa)
    # Bprime: pressure derivative of the bulk modulus (dimensionless)
    return (
        3
        / 2
        * B0
        * (Eta(V, V0) ** 7 - Eta(V, V0) ** 5)
        * (1 + 3 / 4 * (Bprime - 4) * (Eta(V, V0) ** 2 - 1))
    )


def ThirdBMPc(V, V0, B0, Bprime, Pc):
    # The third-order Birch-Murnaghan fit corresponding the equation of state
    # V: unit-cell volume at a pressure point (Angstrom^3)
    # V0: the zero pressure unit-cell volume (Angstrom^3)
    # B0: Bulk modulus at zero pressure (GPa)
    # Bprime: pressure derivative of the bulk modulus (dimensionless)
    # Pc: critical pressure (GPa)
    return (Eta(V, V0) ** 5) * (
        Pc
        - 1 / 2 * ((3 * B0) - (5 * Pc)) * (1 - (Eta(V, V0) ** 2))
        + (9 / 8)
        * B0
        * (Bprime - 4 + (35 * Pc) / (9 * B0))
        * (1 - (Eta(V, V0) ** 2)) ** 2
    )


def WrapperThirdBMPc(InpPc):
    # Allows ThirdBMPc to be fitted using curve_fit() with InpPc as a constant
    # InpPc: input critical pressure (GPa)
    def TempFunc(V, V0, B0, Bprime, Pc=InpPc):
        return ThirdBMPc(V, V0, B0, Bprime, Pc)

    return TempFunc


def NormCRAX(CalCrax, PrinComp):
    # Normalise the crystallographic axes for the indicatrix plot
    # CalCrax: calculated crystallogrphic axes
    # PrinComp: 3 principal components which are
    # coefficient of thermal expansion (MK^-1) or
    # median compressibilities (TPa^-1)
    NormCrax = np.zeros((3, 3))
    maxalpha = np.abs(max(PrinComp[0], PrinComp[1], PrinComp[2]))
    lens = np.zeros(3)
    for i in range(0, 3):
        lenIn = 0
        for j in range(0, 3):
            lenIn += CalCrax[i][j] ** 2
        lens[i] = lenIn ** 0.5
    maxlen = max(lens)

    for i in range(0, 3):  # normalise the axes
        NormCrax[i] = CalCrax[i] * maxalpha / maxlen
    return NormCrax


def Indicatrix(PrinComp):
    # Indicatrix plot for coefficient of thermal expansion (MK^-1) or median compressibilities (TPa^-1)
    # PrinComp: 3 principal components which are
    # coefficient of thermal expansion (MK^-1) or
    # median compressibilities (TPa^-1)
    theta, phi = np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 2 * 100)
    THETA, PHI = np.meshgrid(theta, phi)
    maxIn = np.amax(np.abs(PrinComp))
    R = (
        PrinComp[0] * (np.sin(THETA) * np.cos(PHI)) ** 2
        + PrinComp[1] * (np.sin(THETA) * np.sin(PHI)) ** 2
        + PrinComp[2] * (np.cos(THETA) ** 2)
    )
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    return maxIn, R, X, Y, Z
