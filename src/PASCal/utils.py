import numpy as np
from typing import Union, Callable
from functools import partial


def round_array(var: np.ndarray, dec: int) -> Union[np.ndarray, float]:
    """Rounding the number to desired decimal places
    `round()` is more accurate at rounding float numbers than `np.round()`.

    Parameters:
        var: The array or scalar to round.
        dec: The number of decimal places to round to.

    Returns:
        The rounded array or scalar.

    """
    if var.ndim == 0:
        return round(var, dec)  # type: ignore

    for indices, val in np.ndenumerate(var):
        var[indices] = round(val, dec)
    return var


def orthomat(latt: np.ndarray) -> np.ndarray:
    """Compute the corresponding change-of-basis transformation
    (square matrix M) in E = M x A.

    Parameters:
        latt: The lattice parameters (a, b, c, α, β, γ) in Å and degrees, respectively.

    Returns:
        The change-of-basis matrix M.

    """
    orth = np.zeros((np.shape(latt)[0], 3, 3))
    a, b, c = latt[:3]
    alpha, beta, gamma = np.radians(latt[3:])
    gaS = np.arccos(
        (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))
    )
    orth[0, 0] = 1 / (a * np.sin(beta) * np.sin(gaS))
    orth[1, 0] = np.cos(gaS) / (b * np.sin(alpha) * np.sin(gaS))
    orth[2, 0] = (
        np.cos(alpha) * np.cos(gaS) / np.sin(alpha) + np.cos(beta) / np.sin(beta)
    ) / (-1 * b * np.sin(gaS))
    orth[1, 1] = 1 / (b * np.sin(alpha))
    orth[2, 1] = -1 * np.cos(alpha) / (c * np.sin(alpha))
    orth[2, 2] = 1 / c

    return orth


def cell_vol(latt: np.ndarray) -> float:
    """Calculates the unit-cell volume.

    Parameters:
        latt: The lattice parameters (a, b, c, α, β, γ) in Å and degrees, respectively.

    Returns:
        The unit cell volume in Å³.
    """
    a, b, c = latt[:3]
    alpha, beta, gamma = np.radians(latt[3:])
    return (
        a * b * c
        * (1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2
        + (2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)))** (0.5)
    )


def EmpEq(TP: np.ndarray, Epsilon0: np.ndarray, lambdaP: float, Pc: float, Nu: float) -> np.ndarray:
    """Empirical fit for pressure input data.

    Parameters:
        TP: array of pressure data points,
        Epsilon0: strain at critical pressure
        lambdaP: compressibility in (GPa^-nu)
        Pc: critical pressure (GPa)
        Nu: rate of stiffening (0.5)

    Returns:
        The strain at each pressure point.

    """
    return Epsilon0 + (lambdaP * ((TP - Pc) ** Nu))


def Comp(TP: np.ndarray, lambdaP: float, Pc: float, Nu: float) -> np.ndarray:
    """Calculate the compressibility from the derivative -de/dp.

    Parameters:
        TP: array of pressure data points,
        lambdaP: compressibility in (GPa^-nu)
        Pc: critical pressure (GPa)
        Nu: rate of stiffening (0.5)

    Returns:
        The compressibility at each pressure point.

    """

    return -lambdaP * Nu * ((TP - Pc) ** (Nu - 1))


def CompErr(Pcov: np.ndarray, TP: np.ndarray, lambdaP: float, Pc: float, Nu: float) -> np.ndarray:
    """Calculate errors in compressibilities.

    Parameters:
        Pcov: the estimated covariance of optimal values of the empirical parameters

    Returns:
        The error in compressibility at each pressure point.

    """

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


def Eta(V: np.ndarray, V0: float) -> float:
    """Defining the parameter to be used in Birch-Murnaghan equations of state.

    Parameters:
        V: unit-cell volume at a pressure point in Å³.
        V0: the zero pressure unit-cell volume in Å³.

    Returns:
        The η parameter to be used in Birch-Murnaghan equations of state.
    """
    return np.abs(V0 / V) ** (1 / 3)


def SecBM(V: np.ndarray, V0: float, B: float):
    """The second-order Birch-Murnaghan fit corresponding the equation of state

       P(V) = (3 B / 2) [η⁷ - η⁵]

    Parameters:
        V: unit-cell volume at a pressure point in Å³.
        V0: the zero pressure unit-cell volume in Å³.
        B: Bulk modulus in GPa.

    Returns:
        The second-order P(V) fit at each measured pressure point.

    """
    return (3 / 2) * B * (Eta(V, V0) ** 7 - Eta(V, V0) ** 5)


def ThirdBM(V: np.ndarray, V0: float, B0: float, Bprime: float):
    """The third-order Birch-Murnaghan fit corresponding to the equation of state.

       P(V) = (3 B0 / 2) [η⁷ - η⁵] * [1 + (3(Bprime - 4)/4)[η² - 1]]

    Parameters:
        V: unit-cell volume at a pressure point in Å³.
        V0: the zero pressure unit-cell volume in Å³.
        B0: Bulk modulus in GPa.
        Bprime: pressure derivative of the bulk modulus (GPa/Å³).

    Returns:
        The third-order P(V) fit at each measured pressure point.

    """
    return (
        3
        / 2
        * B0
        * (Eta(V, V0) ** 7 - Eta(V, V0) ** 5)
        * (1 + 3 / 4 * (Bprime - 4) * (Eta(V, V0) ** 2 - 1))
    )


def ThirdBMPc(V: np.ndarray, V0: float, B0: float, Bprime: float, Pc: float):
    """The third-order Birch-Murnaghan fit corresponding the equation of state
    with incorporation of non-zero critical pressure.

        P(V) = η⁵ * [Pc - 1/2(3B0 - 5Pc)(1 - η²) + 9/8B0(Bprime - 4 + 35Pc/9B0)(1 - η²)²]

    from Sata et al, 10.1103/PhysRevB.65.104114 (2002) and Eq. 11
    from Cliffe & Goodwin https://doi.org/10.1107/S0021889812043026 (2012).

    Parameters:
        V: unit-cell volume at a pressure point in Å³.
        V0: the zero pressure unit-cell volume in Å³.
        B0: Bulk modulus in GPa.
        Bprime: pressure derivative of the bulk modulus (GPa/Å³).
        Pc: critical pressure (GPa)

    Returns:
        The third-order P(V) fit at each measured pressure point.
    """
    return (Eta(V, V0) ** 5) * (
        Pc
        - 1 / 2 * ((3 * B0) - (5 * Pc)) * (1 - (Eta(V, V0) ** 2))
        + (9 / 8)
        * B0
        * (Bprime - 4 + (35 * Pc) / (9 * B0))
        * (1 - (Eta(V, V0) ** 2)) ** 2
    )



def WrapperThirdBMPc(InpPc: float) -> Callable:
    """Wrapper for ThirdBMPc to allow it to be used with curve_fit().

    Parameters:
        InpPc: input critical pressure (GPa)
    """
    return partial(ThirdBMPc, Pc=InpPc)


def NormCRAX(CalCrax: np.ndarray, PrinComp: np.ndarray) -> np.ndarray:
    """Normalise the crystallographic axes for the indicatrix plot

    Parameters:
        CalCrax: calculated crystallogrphic axes
        PrinComp: eigenvalues

    Returns:
        Normalised crystallographic axes

    """
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


def Indicatrix(PrinComp: np.ndarray):
    """Indicatrix plot

    Parameters:
        PrinComp: eigenvalues

    Returns:
        Data required for the indicatrix plot

    """
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