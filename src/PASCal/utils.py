import numpy as np
from typing import Union, Tuple


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


def orthomat(lattices: np.ndarray) -> np.ndarray:
    """Compute the corresponding change-of-basis transformations
    (square matrix M) in E = M x A.

    Parameters:
        lattices: An Nx6 array of lattice parameters (a, b, c, α, β, γ)
            in Å and degrees, respectively.

    Returns:
        An array of change-of-basis matrices M for the N cells.

    """
    orth = np.zeros((np.shape(lattices)[0], 3, 3))
    alphas, betas, gammas = (
        np.radians(lattices[:, 3]),
        np.radians(lattices[:, 4]),
        np.radians(lattices[:, 5]),
    )
    gaS = np.arccos(
        (np.cos(alphas) * np.cos(betas) - np.cos(gammas))
        / (np.sin(alphas) * np.sin(betas))
    )
    orth[:, 0, 0] = 1 / (lattices[:, 0] * np.sin(betas) * np.sin(gaS))
    orth[:, 1, 0] = np.cos(gaS) / (lattices[:, 1] * np.sin(alphas) * np.sin(gaS))
    orth[:, 2, 0] = (
        np.cos(alphas) * np.cos(gaS) / np.sin(alphas) + np.cos(betas) / np.sin(betas)
    ) / (-1 * lattices[:, 1] * np.sin(gaS))
    orth[:, 1, 1] = 1 / (lattices[:, 1] * np.sin(alphas))
    orth[:, 2, 1] = -1 * np.cos(alphas) / (lattices[:, 2] * np.sin(alphas))
    orth[:, 2, 2] = 1 / lattices[:, 2]

    return orth


def cell_vols(lattices: np.ndarray) -> np.ndarray:
    """Calculates the unit-cell volumes of a series of N unit cells.

    Parameters:
        lattices: An N x 6 array of lattice parameters (a, b, c, α, β, γ)
            in Å and degrees, respectively.

    Returns:
        The unit cell volumes in Å³.
    """
    alphas, betas, gammas = (
        np.radians(lattices[:, 3]),
        np.radians(lattices[:, 4]),
        np.radians(lattices[:, 5]),
    )
    return np.product(lattices[:, :3], axis=1) * np.sqrt(
        (1 - np.cos(alphas) ** 2 - np.cos(betas) ** 2 - np.cos(gammas) ** 2)
        + (2 * np.cos(alphas) * np.cos(betas) * np.cos(gammas))
    )


def empirical_p_T(
    p_T: np.ndarray, ε_0: np.ndarray, λ_P: float, p_c: float, ν: float
) -> np.ndarray:
    """Empirical fit for pressure input data.

    Parameters:
        p_T: p(T) array of pressure data points,
        ε_0: strain at critical pressure
        λ_P: compressibility in (GPa^-nu)
        p_c: critical pressure (GPa)
        ν: rate of stiffening (0.5)

    Returns:
        The strain at each pressure point.

    """
    return ε_0 + (λ_P * ((p_T - p_c) ** ν))


def compressibility(p_T: np.ndarray, λ_P: float, p_c: float, ν: float) -> np.ndarray:
    """Calculate the compressibility from the derivative -(dε/dp)_T.

    Parameters:
        p_T: p(T) array of pressure data points,
        λ_P: compressibility in (GPa^-nu)
        p_c: critical pressure (GPa)
        ν: rate of stiffening (0.5)

    Returns:
        The compressibility at each pressure point.

    """

    return -λ_P * ν * ((p_T - p_c) ** (ν - 1))


def compressibility_errors(
    p_cov: np.ndarray, p_T: np.ndarray, λ_P: float, p_c: float, ν: float
) -> np.ndarray:
    """Calculate errors in compressibilities.

    Parameters:
        p_cov: the estimated covariance of optimal values of the empirical parameters
        p_T: p(T) array of pressure data points (,
        λ_P: compressibility in (GPa^-nu)
        p_c: critical pressure (GPa)
        ν: rate of stiffening (0.5)

    Returns:
        The error in compressibility at each pressure point.

    """

    J = np.zeros((4, p_T.shape[0]))  # jacobian matrix
    J[1] = ((p_T - p_c) ** (ν - 1)) * ν
    J[2] = -1 * λ_P * ν * (ν - 1) * (p_T - p_c) ** (ν - 2)
    J[3] = (ν * np.log(p_T - p_c) + 1) * ((p_T - p_c) ** (ν - 1)) * λ_P
    return np.sqrt(np.sum(np.dot(J, p_cov) * J, axis=0))


def eta(V: np.ndarray, V_0: float) -> float:
    """Defining parameter to be used in Birch-Murnaghan equations of state.

    Parameters:
        V: unit-cell volume at a pressure point in Å³.
        V_0: the zero pressure unit-cell volume in Å³.

    Returns:
        The η parameter to be used in Birch-Murnaghan equations of state.
    """
    return np.abs(V_0 / V) ** (1 / 3)


def birch_murnaghan_2nd(V: np.ndarray, V_0: float, B: float):
    """The second-order Birch-Murnaghan fit corresponding the equation of state:

       p(V) = (3 B / 2) [η⁷ - η⁵]

    Parameters:
        V: unit-cell volume at a pressure point in Å³.
        V_0: the zero pressure unit-cell volume in Å³.
        B: Bulk modulus in GPa.

    Returns:
        The second-order p(V) fit at each measured pressure point.

    """
    return (3 / 2) * B * (eta(V, V_0) ** 7 - eta(V, V_0) ** 5)


def birch_murnaghan_3rd(V: np.ndarray, V_0: float, B_0: float, Bprime: float):
    """The third-order Birch-Murnaghan fit corresponding to the equation of state:

       p(V) = (3 B_0 / 2) [η⁷ - η⁵] * [1 + (3(Bprime - 4)/4)[η² - 1]]

    Parameters:
        V: unit-cell volume at a pressure point in Å³.
        V_0: the zero pressure unit-cell volume in Å³.
        B_0: Bulk modulus in GPa.
        Bprime: pressure derivative of the bulk modulus (GPa/Å³).

    Returns:
        The third-order p(V) fit at each measured pressure point.

    """
    return (
        3
        / 2
        * B_0
        * (eta(V, V_0) ** 7 - eta(V, V_0) ** 5)
        * (1 + 3 / 4 * (Bprime - 4) * (eta(V, V_0) ** 2 - 1))
    )


def birch_murnaghan_3rd_pc(
    V: np.ndarray, V_0: float, B_0: float, Bprime: float, p_c: float
):
    """The third-order Birch-Murnaghan fit corresponding the equation of state
    with incorporation of non-zero critical pressure:

        p(V) = η⁵ * [p_c - 1/2(3B_0 - 5 p_c)(1 - η²) + (9/8) B_0 (B' - 4 + 35 p_c/(9 B_0))(1 - η²)²]

    from Sata et al, 10.1103/PhysRevB.65.104114 (2002) and Eq. 11
    from Cliffe & Goodwin https://doi.org/10.1107/S0021889812043026 (2012).

    Parameters:
        V: unit-cell volume at a pressure point in Å³.
        V_0: the zero pressure unit-cell volume in Å³.
        B_0: Bulk modulus in GPa.
        Bprime: pressure derivative of the bulk modulus (GPa/Å³).
        p_c: critical pressure (GPa)

    Returns:
        The third-order p(V) fit at each measured pressure point.
    """
    return (eta(V, V_0) ** 5) * (
        p_c
        - 1 / 2 * ((3 * B_0) - (5 * p_c)) * (1 - (eta(V, V_0) ** 2))
        + (9 / 8)
        * B_0
        * (Bprime - 4 + (35 * p_c) / (9 * B_0))
        * (1 - (eta(V, V_0) ** 2)) ** 2
    )


def normalise_crys_axes(
    calc_crys_ax: np.ndarray, principal_components: np.ndarray
) -> np.ndarray:
    """Normalise the crysallographic axes for the indicatrix plot

    Parameters:
        calc_crys_ax: calculated crysallographic axes
        princical_components: eigenvalues

    Returns:
        Normalised crysallographic axes

    """
    maxalpha = np.abs(np.max(principal_components))
    maxlen = np.max(np.linalg.norm(calc_crys_ax, axis=-1))
    return calc_crys_ax * maxalpha / maxlen


def indicatrix(
    principal_components: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate angular data for indicatrix plot.

    Parameters:
        principal_components: eigenvalues

    Returns:
        Index of the maximum prinicpal component, and
            R, X, Y, Z coordinates for the indicatrix plot.

    """
    theta, phi = np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 2 * 100)
    Θ, Φ = np.meshgrid(theta, phi)
    max_index = np.amax(np.abs(principal_components))
    R = (
        principal_components[0] * (np.sin(Θ) * np.cos(Φ)) ** 2
        + principal_components[1] * (np.sin(Θ) * np.sin(Φ)) ** 2
        + principal_components[2] * (np.cos(Θ) ** 2)
    )
    X = R * np.sin(Θ) * np.cos(Φ)
    Y = R * np.sin(Θ) * np.sin(Φ)
    Z = R * np.cos(Θ)
    return max_index, R, X, Y, Z
